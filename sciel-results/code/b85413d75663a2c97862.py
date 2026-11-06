"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.hermite import hermgauss
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    
    # Parameters from contract description
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise
    
    # Time limit guard
    TIME_LIMIT = 110.0  # Leave 10s buffer for evaluation overhead
    
    # Grid for log(S)
    # S range [5, 200] -> log(S) range [1.6, 5.3]
    # Use a dense grid to capture the kink accurately
    n_grid = 1000
    log_s_min = np.log(5)
    log_s_max = np.log(200)
    log_s_grid = np.linspace(log_s_min, log_s_max, n_grid)
    s_grid = np.exp(log_s_grid)
    
    # Gauss-Hermite quadrature nodes and weights for N(0,1)
    # We need to integrate E[ f(S_{t+dt}) | S_t ]
    # S_{t+dt} = S_t * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
    # Let X = log(S_{t+dt}) = log(S_t) + (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    # Z ~ N(0,1)
    # We use Hermite quadrature: integral f(x) exp(-x^2) dx = sum w_i f(x_i)
    # Standard normal integral: (1/sqrt(pi)) integral f(x) exp(-x^2) dx
    # So E[g(Z)] = (1/sqrt(pi)) sum w_i g(x_i) where x_i, w_i are hermgauss nodes
    n_quad = 50
    x_h, w_h = hermgauss(n_quad)
    
    # Precompute the shift and scale for the log-price transition
    mu_log = (r - 0.5 * sigma**2) * dt
    sigma_log = sigma * np.sqrt(dt)
    
    # Backward induction
    # V_k is the value at exercise date k (0 to n_exercise-1)
    # At the final date (k = n_exercise-1), V = max(K - S, 0)
    # For earlier dates, V = max(K - S, exp(-r*dt) * E[V_{k+1} | S])
    
    # We store values on the grid for the next step
    # V_next[i] corresponds to s_grid[i]
    
    # Final step: European put payoff
    V_next = np.maximum(K - s_grid, 0.0)
    
    # Store thresholds for the exercise policy
    # We need to return a policy that takes current state and time.
    # The exercise dates are t_1, ..., t_12.
    # Let's index steps 0 to 11 corresponding to t_1 to t_12.
    # At step j (0-indexed), we are at time t_{j+1}.
    # We decide whether to exercise.
    # If we exercise, we get K - S.
    # If we continue, we get exp(-r*dt) * E[V_{j+2} | S] (for j < 11)
    # Or if j=11, it's the last date, so we must exercise if in the money? 
    # Actually, the problem says "12 monthly exercise dates".
    # Usually, at the last date, the option expires, so it's European-like.
    # The value at the last date is max(K-S, 0).
    # The decision at date j is: Exercise if K - S >= exp(-r*dt) * E[V_{j+1} | S]
    
    # Let's refine the grid and loop
    # We need to compute the continuation value for each grid point at step j
    # based on V_next (which is the value at step j+1)
    
    # To speed up, we can precompute the quadrature points in log-space
    # Z_i are the standard normal quadrature points
    # log(S_{next}) = log(S_current) + mu_log + sigma_log * Z_i
    # We need to interpolate V_next at these new log(S) points.
    
    # Since s_grid is uniform in log(S), we can use linear interpolation on log(S).
    # log_s_grid is the x-axis.
    
    # Precompute weights for the expectation
    # E[ V(S_next) ] = (1/sqrt(pi)) * sum w_i V( S_current * exp(mu_log + sigma_log * x_i) )
    # Note: hermgauss returns x_i, w_i such that integral_{-inf}^{inf} f(x) exp(-x^2) dx = sum w_i f(x_i)
    # E[f(Z)] = (1/sqrt(pi)) sum w_i f(x_i)
    
    weights_norm = w_h / np.sqrt(np.pi)
    
    # We will store the exercise boundary (threshold S) for each of the 12 dates.
    # If S < threshold, exercise (for a put).
    # Since the put payoff is decreasing in S, and continuation value is also decreasing (but less steeply usually),
    # there is a single threshold S* such that for S < S*, we exercise.
    
    thresholds = np.zeros(n_exercise)
    
    # V_next is for the last date (date 12).
    # Loop backwards from date 11 (index 10) down to date 1 (index 0).
    # Wait, the dates are t_1, ..., t_12.
    # V_12 = Payoff.
    # Decision at t_11: Exercise if K - S >= exp(-r*dt) * E[V_12 | S]
    # ...
    # Decision at t_1: Exercise if K - S >= exp(-r*dt) * E[V_2 | S]
    
    # We have V_next = V_12.
    # We need to compute V_11, V_10, ..., V_1.
    
    # Let's iterate j from 10 down to 0 (corresponding to dates 11 down to 1)
    # V_current will be the value at date j+1.
    
    for j in range(n_exercise - 1, -1, -1):
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            # Return a default policy (e.g., exercise if in the money)
            # This is a fallback
            class FallbackPolicy:
                def __init__(self, K):
                    self.K = K
                def exercise(self, current_state, time):
                    # Exercise if S < K
                    return (current_state < self.K).astype(bool)
            return FallbackPolicy(K)
            
        # Compute continuation value on the grid
        # For each S_i in s_grid:
        # log_S_next_i_k = log_S_i + mu_log + sigma_log * x_k
        # S_next_i_k = exp(log_S_next_i_k)
        # V_next_interp_i_k = interpolate(V_next, log_s_grid, log_S_next_i_k)
        
        # Vectorized interpolation
        # log_s_grid shape: (n_grid,)
        # x_h shape: (n_quad,)
        # We want a matrix of shape (n_grid, n_quad) of log_S_next values
        log_s_grid_col = log_s_grid[:, np.newaxis] # (n_grid, 1)
        log_s_next_matrix = log_s_grid_col + mu_log + sigma_log * x_h[np.newaxis, :] # (n_grid, n_quad)
        
        # Interpolate V_next
        # np.interp expects xp to be increasing. log_s_grid is increasing.
        # We need to handle values outside the grid range.
        # For a put, if S is very high, V=0. If S is very low, V approx K.
        # Let's use linear interpolation with left/right filling.
        # Since log_s_next_matrix might be outside [log_s_min, log_s_max],
        # we clip or let np.interp handle it (it fills with endpoints).
        # For a put, V(S) -> K as S->0, V(S)->0 as S->inf.
        # The grid goes from 5 to 200.
        # If log_S_next < log(5), S_next < 5. V should be close to K.
        # If log_S_next > log(200), S_next > 200. V should be 0.
        
        # np.interp(x, xp, fp, left=fp[0], right=fp[-1])
        # fp[0] is V(5), fp[-1] is V(200).
        
        V_next_interp = np.interp(log_s_next_matrix, log_s_grid, V_next, left=K, right=0.0)
        
        # Expected value: sum over quadrature points
        # E = sum_k weights_norm[k] * V_next_interp[:, k]
        E_V_next = np.dot(V_next_interp, weights_norm) # shape (n_grid,)
        
        # Discounted continuation value
        disc_factor = np.exp(-r * dt)
        cont_value = disc_factor * E_V_next
        
        # Immediate exercise value
        ex_value = np.maximum(K - s_grid, 0.0)
        
        # Optimal value
        V_current = np.maximum(ex_value, cont_value)
        
        # Determine threshold
        # Exercise if ex_value >= cont_value
        # This is true for S < S*
        # We find the largest S in the grid where ex_value >= cont_value
        # Or simply find the boundary.
        # Since ex_value is decreasing and cont_value is decreasing,
        # the set of S where we exercise is an interval [0, S*].
        
        # Find indices where we exercise
        exercise_mask = ex_value >= cont_value
        if np.any(exercise_mask):
            # Find the last index where we exercise
            # s_grid is increasing.
            idx_last_ex = np.max(np.where(exercise_mask)[0])
            # Interpolate to find the exact threshold?
            # For simplicity, use the grid point.
            # Or linear interpolate between idx_last_ex and idx_last_ex+1
            if idx_last_ex < n_grid - 1:
                # Linear interpolation for the boundary
                S1 = s_grid[idx_last_ex]
                S2 = s_grid[idx_last_ex + 1]
                V1_ex = ex_value[idx_last_ex]
                V1_c = cont_value[idx_last_ex]
                V2_ex = ex_value[idx_last_ex + 1]
                V2_c = cont_value[idx_last_ex + 1]
                
                diff1 = V1_ex - V1_c
                diff2 = V2_ex - V2_c
                if diff1 != diff2:
                    frac = diff1 / (diff1 - diff2)
                    S_star = S1 + frac * (S2 - S1)
                else:
                    S_star = S1
            else:
                S_star = s_grid[idx_last_ex]
            thresholds[j] = S_star
        else:
            # Never exercise? (Unlikely for a put with these params)
            thresholds[j] = 0.0
        
        # Prepare for next iteration
        V_next = V_current
    
    # The thresholds array is indexed by j (0 to 11) corresponding to dates 1 to 12.
    # j=0 is date 1 (t=1/12), j=11 is date 12 (t=1).
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_exercise = len(thresholds)
            self.dt = T / self.n_exercise
            
        def exercise(self, current_state, time):
            # current_state: array of S values
            # time: float, current time
            # Determine which exercise date we are at
            # The exercise dates are at t_k = k * dt for k=1..12
            # If time is close to a date, use that date's threshold.
            # The evaluator calls exercise(current_state, time) at the exercise dates.
            
            # Map time to index
            # t_k = k * dt
            # k = round(time / dt)
            # Clamp k to 1..12
            k = int(np.round(time / self.dt))
            if k < 1:
                k = 1
            elif k > self.n_exercise:
                k = self.n_exercise
            
            # Index in thresholds array is k-1
            idx = k - 1
            S_star = self.thresholds[idx]
            
            # Exercise if S < S_star
            return (current_state < S_star).astype(bool)
            
    return ExercisePolicy(thresholds, K)

# EVOLVE-BLOCK-END
