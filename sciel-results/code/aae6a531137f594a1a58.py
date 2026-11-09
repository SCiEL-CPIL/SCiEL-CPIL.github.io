"""Bermudan put exercise policy: exact grid-based backward induction with smooth logistic boundary."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    # Convert training paths to numpy array (used only to infer shape/params if needed, 
    # but we rely on contract params for the model)
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_times = paths.shape

    # 12 exercise dates (no time-0), so times are 1..12
    # dt = T / n_steps
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Grid setup
    # We need a grid that covers the relevant range of spot prices.
    # For a put with K=100, S0=100, sigma=0.2, T=1:
    # Max up move: 100 * exp(0.05*1 + 0.2*2) ~ 150
    # Min down move: 100 * exp(0.05*1 - 0.2*2) ~ 67
    # But with 12 steps, the range is tighter.
    # Let's use a wide log-spaced grid to be safe.
    S_min = 1.0
    S_max = 300.0
    n_grid = 500
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Precompute quadrature weights for Gauss-Hermite
    # We need to compute E[f(S_next) | S] where S_next = S * exp(drift + vol * Z)
    # Z ~ N(0,1)
    # E[f(S * exp(d + v*Z))] = (1/sqrt(2*pi)) * integral f(S * exp(d + v*z)) * exp(-z^2/2) dz
    # Let u = z/sqrt(2), then z = u*sqrt(2), dz = sqrt(2) du
    # Integral becomes (1/sqrt(2*pi)) * integral f(S * exp(d + v*u*sqrt(2))) * exp(-u^2) * sqrt(2) du
    # = (1/sqrt(pi)) * integral f(S * exp(d + v*u*sqrt(2))) * exp(-u^2) du
    # Gauss-Hermite: integral g(u) exp(-u^2) du ≈ sum w_i g(u_i)
    # So E[...] ≈ (1/sqrt(pi)) * sum w_i f(S * exp(d + v*u_i*sqrt(2)))
    
    n_quad = 32
    u_quad, w_quad = roots_hermite(n_quad)
    # Scale weights by 1/sqrt(pi)
    w_quad_scaled = w_quad / np.sqrt(np.pi)
    
    # Precompute the exponent terms for the transition
    # S_next = S * exp(drift + vol * z)
    # z = u * sqrt(2)
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    
    # transition_factor[i] = exp(drift + vol * u_i * sqrt(2))
    # S_next[:, i] = S_grid[:, None] * transition_factor[i]
    transition_factors = np.exp(drift + vol * u_quad * np.sqrt(2))
    
    # Initialize value function at terminal time (n_steps)
    # V_n(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store optimal thresholds for each exercise date
    # We will store the boundary points (S values) where it is optimal to switch from exercise to continue
    # For a put, it is optimal to exercise for low S and continue for high S.
    # So there is a single threshold T_j such that exercise if S <= T_j.
    
    thresholds = []
    
    # Backward induction from n_steps-1 down to 1
    for j in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 2.0:
            break
        
        # Compute continuation value: disc * E[V_next(S_next) | S]
        # S_next = S_grid[:, None] * transition_factors[None, :]
        S_next = S_grid[:, None] * transition_factors[None, :]
        
        # Interpolate V_next at S_next
        # Since S_next is sorted in each column? No, S_grid is sorted, transition_factors are positive.
        # So S_next[:, i] is sorted for each i.
        # We can use np.interp efficiently.
        # However, np.interp works on 1D arrays. We can reshape.
        V_next_interp = np.interp(S_next.ravel(), S_grid, V_next).reshape(n_grid, n_quad)
        
        # Weighted sum
        cont_value = disc * np.sum(V_next_interp * w_quad_scaled[None, :], axis=1)
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_current = np.maximum(intrinsic, cont_value)
        
        # Determine threshold
        # Exercise if intrinsic >= cont_value
        exercise_mask = intrinsic >= cont_value
        
        # Find the largest S where exercise is optimal
        # Since S_grid is sorted, exercise_mask should be True for low S and False for high S
        # Find the last index where exercise_mask is True
        if exercise_mask.any():
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            # Threshold is between S_grid[last_exercise_idx] and S_grid[last_exercise_idx+1]
            if last_exercise_idx + 1 < n_grid:
                # Use the geometric mean or midpoint? Midpoint in log space is better for GBM
                threshold = np.sqrt(S_grid[last_exercise_idx] * S_grid[last_exercise_idx + 1])
            else:
                threshold = S_grid[last_exercise_idx]
        else:
            threshold = 0.0
            
        thresholds.append(threshold)
        V_next = V_current
        
    # thresholds is from last exercise date to first
    # thresholds[0] corresponds to exercise date n_steps-1
    # thresholds[1] corresponds to exercise date n_steps-2
    # ...
    # thresholds[n_steps-2] corresponds to exercise date 1
    
    # We need thresholds for dates 1..n_steps-1
    # Let's reverse the list
    thresholds_reversed = thresholds[::-1]
    
    # If we broke early, we might have fewer thresholds. Pad with the last one or 0?
    # If we broke early, we assume the policy for earlier dates is similar or we just use what we have.
    # But for safety, if len < n_steps-1, pad with the last computed threshold (which is for the earliest date we computed)
    # Actually, if we broke early, we computed dates n_steps-1, ..., j+1. We are missing 1..j.
    # The thresholds for earlier dates (lower j) are typically higher (more likely to exercise? No, for puts, earlier dates have higher thresholds? 
    # Let's check: as time to maturity decreases, the option value decreases, making exercise more attractive. 
    # So thresholds should increase as we go backward in time (from n_steps-1 to 1).
    # So thresholds_reversed[0] (date 1) should be the highest.
    # If we ran out of time, we have the later dates (lower thresholds). We should probably pad the front with the highest threshold we have?
    # Or just use the available ones and default to "never exercise" or "always exercise" for missing?
    # Given the short time (12 steps), it should finish quickly. 500 grid * 32 quad * 12 steps = 192k ops. Very fast.
    
    if len(thresholds_reversed) < n_steps - 1:
        # Pad with the last threshold (which is the highest, for the earliest date we computed)
        # This is a heuristic fallback
        last_th = thresholds_reversed[-1] if thresholds_reversed else 0.0
        while len(thresholds_reversed) < n_steps - 1:
            thresholds_reversed.insert(0, last_th)
    
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_steps):
            self.thresholds = thresholds  # list of thresholds for dates 1..n_steps-1
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise at the given time.
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            if time <= 0 or time > self.n_steps - 1:
                return np.zeros(len(current_state), dtype=bool)
            
            # time is 1-indexed exercise date
            # thresholds list is for dates 1..n_steps-1
            # Index into thresholds: time-1
            if time - 1 < len(self.thresholds):
                threshold = self.thresholds[time - 1]
            else:
                threshold = 0.0
            
            # Exercise if S <= threshold
            return current_state <= threshold

    # Create policy
    policy = ExercisePolicy(thresholds_reversed, K, r, sigma, T, n_steps)
    return policy
# EVOLVE-BLOCK-END
