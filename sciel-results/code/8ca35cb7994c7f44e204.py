# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_hermite, eval_hermite

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using backward induction with Gauss-Hermite quadrature.
    
    This approach computes the continuation value exactly (up to quadrature precision)
    by exploiting the log-normal transition density of GBM. It avoids the approximation
    errors of regression-based methods (like Longstaff-Schwartz) and grid-based methods
    by using a deterministic quadrature rule that integrates the value function against
    the conditional probability density of the next state.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', contract.get('0.2', 0.2))
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_cols)
    n_paths, n_cols = training_paths.shape
    
    # Determine which columns correspond to exercise dates
    if n_cols > n_dates:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    # Ensure we have exactly n_dates columns
    if exercise_data.shape[1] < n_dates:
        pass

    n_paths_eff, n_steps_eff = exercise_data.shape
    if n_steps_eff != n_dates:
        n_dates = n_steps_eff

    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # Gauss-Hermite quadrature setup
    # We need to integrate E[V(S_{t+1}) | S_t]
    # S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # Let Z ~ N(0,1). We use Gauss-Hermite quadrature for N(0,1).
    # Standard Gauss-Hermite integrates exp(-x^2) f(x).
    # To integrate exp(-0.5 z^2) g(z), we use the change of variable or the scaled version.
    # scipy.special.roots_hermite(n) gives nodes x_i and weights w_i for exp(-x^2).
    # We need to handle the N(0,1) density: 1/sqrt(2*pi) * exp(-0.5 z^2).
    
    n_quad = 32  # Number of quadrature points. 32 is usually very accurate for smooth functions.
    # We can increase this if time permits, but 32 is fast and accurate.
    
    # Get nodes and weights for standard normal distribution
    # scipy.special.roots_hermite returns nodes for exp(-x^2).
    # The relation for N(0,1) is: E[f(Z)] = 1/sqrt(2*pi) * sum w_i * f(x_i / sqrt(2)) * sqrt(2) ?
    # Let's derive carefully.
    # Integral_{-inf}^{inf} exp(-x^2) f(x) dx ≈ sum w_i f(x_i)
    # We want I = Integral_{-inf}^{inf} (1/sqrt(2*pi)) exp(-0.5 z^2) g(z) dz
    # Let z = sqrt(2) * x. Then dz = sqrt(2) dx.
    # exp(-0.5 z^2) = exp(-x^2)
    # I = (1/sqrt(2*pi)) * sqrt(2) * Integral_{-inf}^{inf} exp(-x^2) g(sqrt(2) x) dx
    # I = (1/sqrt(pi)) * Integral_{-inf}^{inf} exp(-x^2) g(sqrt(2) x) dx
    # Using Gauss-Hermite:
    # I ≈ (1/sqrt(pi)) * sum w_i * g(sqrt(2) x_i)
    
    x_gh, w_gh = roots_hermite(n_quad)
    # z_nodes are the values of Z
    z_nodes = x_gh * np.sqrt(2)
    # Weights for the standard normal integral
    # w_normal = w_gh / sqrt(pi)
    w_normal = w_gh / np.sqrt(np.pi)
    
    # Precompute the exponential terms for the transition
    # S_{t+1} = S_t * exp(A + B * Z)
    # A = (r - 0.5*sigma^2)*dt
    # B = sigma*sqrt(dt)
    A = (r - 0.5 * sigma**2) * dt
    B = sigma * np.sqrt(dt)
    
    # exp(B * Z) for each quadrature node
    exp_B_Z = np.exp(B * z_nodes)
    
    # Store the value function at each time step as a list of (S_values, V_values)
    # We will use interpolation to evaluate V_{t+1}(S) at the quadrature points.
    # To make this fast and robust, we can store the sorted states and values.
    
    # V_interp[t] will be a function or data structure to evaluate V_t(S)
    # Since we only evaluate at specific points (quadrature nodes scaled by S_t),
    # and we do this for all paths, we can vectorize the interpolation.
    
    # Strategy:
    # 1. Start from T. V_T(S) = max(K-S, 0).
    # 2. For t from T-1 down to 0:
    #    a. We have V_{t+1} defined on the training states S_{t+1}.
    #    b. We need to compute C_t(S_t) = discount * E[V_{t+1}(S_{t+1}) | S_t].
    #    c. For each path i, S_{t+1,j} = S_{t,i} * exp(A + B * z_j).
    #    d. We need V_{t+1}(S_{t+1,j}).
    #    e. We can pre-sort the training states at t+1 and their values.
    #    f. Use np.interp to evaluate V_{t+1} at the quadrature points for all paths.
    
    # To speed up np.interp, we sort the training data once per time step.
    
    # V_values[t] will store the value at the training paths for time t.
    # S_values[t] will store the states at the training paths for time t.
    
    S_values = [None] * n_dates
    V_values = [None] * n_dates
    
    # Time T (last date, index n_dates - 1)
    S_T = exercise_data[:, -1]
    V_T = np.maximum(K - S_T, 0.0)
    S_values[-1] = S_T
    V_values[-1] = V_T
    
    # Backward induction
    for t in range(n_dates - 2, -1, -1):
        S_t = exercise_data[:, t]
        S_t_next = exercise_data[:, t+1]
        V_t_next = V_values[t+1]
        
        # Sort S_t_next and V_t_next for interpolation
        # We need to sort them together
        sort_idx = np.argsort(S_t_next)
        S_t_next_sorted = S_t_next[sort_idx]
        V_t_next_sorted = V_t_next[sort_idx]
        
        # Compute continuation value for each path i
        # S_next_quad[i, j] = S_t[i] * exp(A + B * z_j)
        # Vectorized computation:
        # S_t shape: (N,)
        # exp_B_Z shape: (M,)
        # S_next_quad shape: (N, M)
        
        # To save memory and time, we can do this in chunks if N is large, 
        # but 4096 paths * 32 quad points = 131k floats, which is small.
        
        S_next_quad = S_t[:, np.newaxis] * np.exp(A) * exp_B_Z[np.newaxis, :]
        
        # Interpolate V_t_next at S_next_quad
        # np.interp expects xp to be 1D.
        # We can use np.interp on each row, but that's a loop.
        # Better: use a vectorized approach or just loop over paths if N is not huge.
        # 4096 paths is small enough for a loop in Python if the inner operation is fast?
        # np.interp is C-speed. A Python loop of 4096 iterations calling np.interp might be slow.
        # Let's try to vectorize.
        
        # Alternative: use scipy.interpolate.interp1d? No, that's object overhead.
        # Let's use the fact that S_next_quad is (N, M).
        # We can use a simple loop over the quadrature points? No.
        
        # Actually, we can use np.searchsorted to find the indices and then interpolate.
        # This is fully vectorizable.
        
        # Find indices for interpolation
        # idx = np.searchsorted(S_t_next_sorted, S_next_quad, side='left')
        # Then clip idx to [1, len-1] to ensure we have left and right neighbors.
        # Then compute the interpolation weights.
        
        idx = np.searchsorted(S_t_next_sorted, S_next_quad, side='left')
        # Clip indices to valid range [1, len(S_t_next_sorted)-1]
        # We need to handle the case where S_next_quad is outside the range of S_t_next_sorted.
        # np.interp handles this by clamping to the endpoints.
        # searchsorted returns 0 if x < xp[0], and len(xp) if x > xp[-1].
        
        # For interpolation, we need i such that xp[i-1] <= x <= xp[i]
        # If idx == 0, we use the first value.
        # If idx == len, we use the last value.
        # Otherwise, we interpolate between idx-1 and idx.
        
        # Let's create arrays for the left and right points.
        # left_idx = np.clip(idx - 1, 0, len(S_t_next_sorted) - 1)
        # right_idx = np.clip(idx, 0, len(S_t_next_sorted) - 1)
        
        # But if idx == 0, left_idx = 0, right_idx = 0. Interpolation is trivial.
        # If idx == len, left_idx = len-1, right_idx = len-1.
        
        # However, we must be careful.
        # Let's use np.interp in a loop for simplicity and reliability, as 4096 is small.
        # Time check: 4096 iterations * 12 steps = 49k calls to np.interp.
        # np.interp is very fast. This should be well within 110s.
        
        C_t = np.zeros_like(S_t)
        
        # To optimize, we can precompute the denominator for interpolation
        # But S_t_next_sorted is fixed for this step.
        
        # Let's try the vectorized searchsorted approach for speed.
        n_sorted = len(S_t_next_sorted)
        
        # Handle boundaries
        # If S_next_quad < S_t_next_sorted[0], value = V_t_next_sorted[0]
        # If S_next_quad > S_t_next_sorted[-1], value = V_t_next_sorted[-1]
        
        # Mask for in-range
        # Actually, np.interp is easier. Let's stick to the loop but make it efficient.
        # Or, we can use a trick with np.interp on the flattened array? No, xp must be 1D.
        
        # Let's use the vectorized interpolation formula.
        # i = np.clip(idx, 1, n_sorted - 1)
        # x0 = S_t_next_sorted[i-1]
        # x1 = S_t_next_sorted[i]
        # y0 = V_t_next_sorted[i-1]
        # y1 = V_t_next_sorted[i]
        # weight = (S_next_quad - x0) / (x1 - x0)
        # V_interp = y0 + weight * (y1 - y0)
        
        # This works if idx is in [1, n_sorted-1].
        # We need to handle idx == 0 and idx == n_sorted separately.
        
        i = np.clip(idx, 1, n_sorted - 1)
        x0 = S_t_next_sorted[i-1]
        x1 = S_t_next_sorted[i]
        y0 = V_t_next_sorted[i-1]
        y1 = V_t_next_sorted[i]
        
        # Avoid division by zero if x1 == x0 (duplicates in sorted states)
        # Add a small epsilon or handle duplicates.
        # If x1 == x0, weight is undefined, but y0 == y1 if data is consistent.
        # Let's assume states are distinct enough or add epsilon.
        denom = x1 - x0
        denom = np.where(denom == 0, 1e-12, denom)
        
        weight = (S_next_quad - x0) / denom
        V_interp = y0 + weight * (y1 - y0)
        
        # Correct for out-of-bounds
        # If idx == 0, V_interp should be V_t_next_sorted[0]
        # If idx == n_sorted, V_interp should be V_t_next_sorted[-1]
        
        V_interp[idx == 0] = V_t_next_sorted[0]
        V_interp[idx == n_sorted] = V_t_next_sorted[-1]
        
        # Compute expectation
        # C_t[i] = discount * sum_j w_normal[j] * V_interp[i, j]
        
        # w_normal shape: (M,)
        # V_interp shape: (N, M)
        # C_t shape: (N,)
        
        C_t = discount * np.dot(V_interp, w_normal)
        
        # Compute Intrinsic Value
        Intrinsic_t = np.maximum(K - S_t, 0.0)
        
        # Compute V_t(S_t) = max(Intrinsic_t, C_t)
        V_t = np.maximum(Intrinsic_t, C_t)
        
        S_values[t] = S_t
        V_values[t] = V_t
        
        # Check time
        if time.time() > deadline:
            # If we run out of time, we have V_values for t..T.
            # We need to fill the rest (0..t-1).
            # We can break and handle None in the class.
            break
            
    # Fill any remaining None S_values/V_values
    for t in range(n_dates):
        if S_values[t] is None:
            # Use the last known values or zeros
            # Find next non-None
            for j in range(t+1, n_dates):
                if S_values[j] is not None:
                    S_values[t] = S_values[j]
                    V_values[t] = V_values[j]
                    break
            else:
                # Fallback: use intrinsic value for all states?
                # We need S_values to be non-None for the class.
                # If no data, we can't do much. But this shouldn't happen.
                S_values[t] = np.zeros(n_paths_eff)
                V_values[t] = np.zeros(n_paths_eff)
                
    # Prepare data for the ExercisePolicy class
    # We need to store the sorted states and values for each time step
    # so that we can perform the same interpolation in the exercise() method.
    
    sorted_data = []
    for t in range(n_dates):
        S_t = S_values[t]
        V_t = V_values[t]
        sort_idx = np.argsort(S_t)
        sorted_data.append((S_t[sort_idx], V_t[sort_idx]))
        
    class ExercisePolicy:
        def __init__(self, sorted_data, K, A, B, exp_B_Z, w_normal, discount):
            self.sorted_data = sorted_data
            self.K = K
            self.A = A
            self.B = B
            self.exp_B_Z = exp_B_Z
            self.w_normal = w_normal
            self.discount = discount
            self.n_quad = len(exp_B_Z)
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            Returns: boolean vector
            """
            if time >= len(self.sorted_data):
                return (self.K - current_state) > 0
                
            S_t_next_sorted, V_t_next_sorted = self.sorted_data[time + 1]
            n_sorted = len(S_t_next_sorted)
            
            # Compute S_next_quad
            S_next_quad = current_state[:, np.newaxis] * np.exp(self.A) * self.exp_B_Z[np.newaxis, :]
            
            # Interpolate
            idx = np.searchsorted(S_t_next_sorted, S_next_quad, side='left')
            i = np.clip(idx, 1, n_sorted - 1)
            
            x0 = S_t_next_sorted[i-1]
            x1 = S_t_next_sorted[i]
            y0 = V_t_next_sorted[i-1]
            y1 = V_t_next_sorted[i]
            
            denom = x1 - x0
            denom = np.where(denom == 0, 1e-12, denom)
            
            weight = (S_next_quad - x0) / denom
            V_interp = y0 + weight * (y1 - y0)
            
            V_interp[idx == 0] = V_t_next_sorted[0]
            V_interp[idx == n_sorted] = V_t_next_sorted[-1]
            
            C_t = self.discount * np.dot(V_interp, self.w_normal)
            
            Intrinsic_t = np.maximum(self.K - current_state, 0.0)
            
            return Intrinsic_t >= C_t
            
    return ExercisePolicy(sorted_data, K, A, B, exp_B_Z, w_normal, discount)

# EVOLVE-BLOCK-END
