"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _european_put_price(S, K, r, sigma, T):
    """Analytic European put price for GBM."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from scipy.stats import norm
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.maximum(put, 0.0)

def _fit_boundary(params, exercise_dates, training_paths, K, r, sigma, T, n_exercise):
    """
    Fit a parametric smooth exercise boundary.
    Boundary is a monotonic function of log-spot at each exercise date.
    We use a polynomial in log(S/K) with coefficients that vary smoothly over time.
    
    params: array of shape (n_coeffs, n_exercise) where each column gives
            coefficients for the boundary at that exercise date.
            Boundary: exercise if log(S/K) < b0 + b1*t + b2*t^2 + ... 
            where t = log(S/K)
    """
    n_paths, n_dates = training_paths.shape
    dt = T / n_exercise
    
    # We'll use backward induction to compute the optimal policy
    # Continuation value at time i: discounted expected value of max(exercise, continue) at time i+1
    # Using European put as initial approximation for continuation
    
    # First, compute the optimal exercise boundary via backward induction
    # State space: use quantiles of training paths for discretization
    n_grid = 200
    all_spots = training_paths[:, 1:]  # exclude time 0
    flat_spots = all_spots.flatten()
    
    # Use log-spaced grid from min to max of training paths
    s_min = np.percentile(flat_spots, 0.1)
    s_max = np.percentile(flat_spots, 99.9)
    s_min = max(s_min, 1.0)
    s_max = min(s_max, 500.0)
    
    # Log-spaced grid
    grid = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_grid))
    
    # Backward induction
    # V[i] = value at exercise date i (0-indexed, so i=0 is first exercise date at t=dt)
    # At each date, we can exercise (get K - S if S < K) or continue (get discounted expected V[i+1])
    
    # Start from last exercise date
    # At last date (i = n_exercise - 1), no continuation, so exercise if K - S > 0
    V = np.zeros((n_exercise + 1, n_grid))  # V[n_exercise] = 0 (terminal)
    
    # Terminal: after all exercise dates, value is 0 (or we could use European put with T=0)
    V[n_exercise, :] = 0.0
    
    for i in range(n_exercise - 1, -1, -1):
        # Exercise value
        exercise_val = np.maximum(K - grid, 0.0)
        
        # Continuation value: discounted expected V[i+1]
        # Under risk-neutral GBM: S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # We need E[V[i+1](S_{t+dt}) | S_t = grid[j]]
        # Use Monte Carlo from training paths or quadrature
        
        # Use Gauss-Hermite quadrature for the expectation
        # Z ~ N(0,1), S_next = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # E[f(S_next)] = sum_k w_k * f(S * exp((r-0.5*sigma^2)*dt + sigma*sqrt(dt)*x_k))
        # where x_k, w_k are Gauss-Hermite nodes and weights
        
        from scipy.special import roots_hermite
        if not hasattr(_fit_boundary, '_gh_nodes'):
            n_gh = 32
            x_k, w_k = roots_hermite(n_gh)
            _fit_boundary._gh_nodes = (x_k, w_k)
        x_k, w_k = _fit_boundary._gh_nodes
        
        # For each grid point, compute continuation value
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        cont_val = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = grid[j]
            # S_next for each quadrature point
            log_S_next = np.log(S_j) + drift + vol * x_k
            S_next = np.exp(log_S_next)
            
            # Interpolate V[i+1] at S_next
            V_next = V[i + 1, :]
            # Clip S_next to grid range
            S_next_clipped = np.clip(S_next, s_min, s_max)
            V_interp = np.interp(S_next_clipped, grid, V_next)
            
            # Gauss-Hermite: E[f(Z)] = sum w_k * f(x_k) / sqrt(pi)
            cont_val[j] = np.exp(-r * dt) * np.sum(w_k * V_interp) / np.sqrt(np.pi)
        
        # Optimal value
        V[i, :] = np.maximum(exercise_val, cont_val)
        
        # Store exercise decision: exercise if exercise_val >= cont_val
        # This defines the boundary
    
    # Now extract the exercise boundary from the backward induction
    # For each exercise date i, find the spot level where exercise_val = cont_val
    boundaries = np.zeros(n_exercise)
    for i in range(n_exercise):
        exercise_val = np.maximum(K - grid, 0.0)
        # Recompute continuation for this date
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        cont_val = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = grid[j]
            log_S_next = np.log(S_j) + drift + vol * x_k
            S_next = np.exp(log_S_next)
            S_next_clipped = np.clip(S_next, s_min, s_max)
            V_interp = np.interp(S_next_clipped, grid, V[i + 1, :])
            cont_val[j] = np.exp(-r * dt) * np.sum(w_k * V_interp) / np.sqrt(np.pi)
        
        # Find where exercise_val - cont_val changes sign
        diff = exercise_val - cont_val
        # Exercise when diff > 0, i.e., S < boundary
        # Find the crossing point
        if diff[0] <= 0:
            boundaries[i] = s_min  # Always exercise
        elif diff[-1] >= 0:
            boundaries[i] = s_max  # Never exercise (or very high boundary)
        else:
            # Find sign change
            sign_change = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_change) > 0:
                idx = sign_change[0]
                # Linear interpolation
                x0, x1 = grid[idx], grid[idx + 1]
                y0, y1 = diff[idx], diff[idx + 1]
                if y1 != y0:
                    boundaries[i] = x0 - y0 * (x1 - x0) / (y1 - y0)
                else:
                    boundaries[i] = x0
            else:
                boundaries[i] = K  # Default
    
    # Now fit a smooth parametric boundary
    # Model: boundary_i = f(t_i) where t_i = i / n_exercise (time fraction)
    # Use a smooth function, e.g., polynomial in t
    # Or use the boundaries directly with smoothing
    
    # For the exercise policy, we'll use the fitted boundaries
    # Exercise if S < boundary at the current exercise date
    
    return boundaries, grid, V

class ExercisePolicy:
    def __init__(self, boundaries, contract):
        self.boundaries = boundaries
        self.contract = contract
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.n_exercise = contract.get('n_exercise', 12)
        self.dt = self.T / self.n_exercise
    
    def exercise(self, current_state, time):
        """
        current_state: array of spot prices
        time: current time (or exercise date index)
        Returns: boolean array, True if exercise
        """
        # Determine which exercise date we're at
        # time is in [0, T], exercise dates are at dt, 2*dt, ..., T
        # Find the closest exercise date index
        if time <= 0:
            # No exercise at time 0
            return np.zeros(len(current_state), dtype=bool)
        
        # Exercise date index (0-based)
        date_idx = int(round(time / self.dt)) - 1
        date_idx = max(0, min(date_idx, self.n_exercise - 1))
        
        boundary = self.boundaries[date_idx]
        
        # Exercise if spot < boundary
        return current_state < boundary

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s buffer
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    # Ensure we have the right shape: (n_paths, n_exercise)
    # training_paths should have prices at each exercise date
    n_paths, n_dates = paths.shape
    
    if n_dates < n_exercise:
        # Pad or adjust
        n_exercise = n_dates
    
    # Fit the boundary using backward induction
    boundaries, grid, V = _fit_boundary(None, None, paths, K, r, sigma, T, n_exercise)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        # Return a simple threshold policy
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time):
                return current_state < self.K
        return SimplePolicy(K)
    
    return ExercisePolicy(boundaries, contract)
# EVOLVE-BLOCK-END
