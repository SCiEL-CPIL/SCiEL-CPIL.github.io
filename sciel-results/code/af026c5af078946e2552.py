"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses backward induction on a fine grid to compute optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Monthly exercise dates
    dt = T / n_exercises
    
    # Build a fine grid for the stock price
    # The stock follows GBM: S_t = S_0 * exp((r - sigma^2/2)*t + sigma*W_t)
    # For a put option with K=100, relevant range is roughly [0, 2*K]
    # Use a log-spaced grid for better resolution near zero
    n_grid = 2000
    S_min = 1e-6
    S_max = 2.5 * K  # generous upper bound
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Discount factor
    disc = np.exp(-r * dt)
    
    # Risk-neutral probabilities (using drift-adjusted GBM)
    # Up and down factors for GBM over dt:
    # mu = r - sigma^2/2
    # sigma_dt = sigma * sqrt(dt)
    # In log space: log(S_{t+dt}/S_t) ~ N((r-sigma^2/2)*dt, sigma^2*dt)
    # For a 2-point approximation (like binomial but continuous):
    # We'll use the exact transition probabilities for GBM
    
    mu_dt = (r - 0.5 * sigma**2) * dt
    sigma_dt = sigma * np.sqrt(dt)
    
    # For each grid point S_i, compute E[S_{t+dt} * f(S_{t+dt})] given S_t = S_i
    # S_{t+dt} = S_i * exp(mu_dt + sigma_dt * Z) where Z ~ N(0,1)
    # We need to integrate over the grid
    
    # Precompute: for each grid point, what are the expected continuation values?
    # We do backward induction: V_j(S) = max(K - S, disc * E[V_{j+1}(S_{t+dt}) | S_t = S])
    
    # At maturity (last exercise date, j = n_exercises-1):
    # V_{n_exercises}(S) = max(K - S, 0)  (put payoff)
    
    # Start from the last exercise date
    V = np.maximum(K - S_grid, 0.0)
    
    # Boundaries for each exercise date (index 0 to n_exercises-1)
    boundaries = np.zeros(n_exercises)
    
    # For the last exercise date, boundary is where K - S > 0, i.e., S < K
    # But we want the optimal boundary, which for the last date is simply K (exercise if in the money)
    boundaries[n_exercises - 1] = K
    
    # Now work backwards from j = n_exercises-2 down to 0
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline:
            # Emergency fallback: use current boundaries for remaining dates
            for k in range(j, -1, -1):
                boundaries[k] = K
            break
        
        # Compute continuation value: disc * E[V(S_{t+dt}) | S_t = S_i]
        # S_{t+dt} = S_i * exp(mu_dt + sigma_dt * Z)
        # We approximate the expectation by interpolating V on the grid
        
        # For each S_i, the log of S_{t+dt} is log(S_i) + mu_dt + sigma_dt * Z
        # We need to compute E[V(S_i * exp(mu_dt + sigma_dt * Z))]
        
        # Use a quadrature approach: discretize Z into several points
        # Use 32-point Gauss-Hermite or simple uniform quadrature
        n_quad = 32
        # Use standard normal quadrature points
        # For simplicity, use a symmetric grid
        z_quad = np.linspace(-4.0, 4.0, n_quad)
        # Weights for uniform quadrature over [-4, 4]
        dz = (4.0 - (-4.0)) / (n_quad - 1)
        # Approximate normal PDF at each point
        pdf_vals = np.exp(-0.5 * z_quad**2) / np.sqrt(2 * np.pi)
        weights = pdf_vals * dz
        weights = weights / np.sum(weights)  # normalize
        
        # For each S_i, compute S_{t+dt} = S_i * exp(mu_dt + sigma_dt * z_quad)
        # Shape: (n_grid, n_quad)
        log_S_next = np.log(S_grid)[:, np.newaxis] + mu_dt + sigma_dt * z_quad[np.newaxis, :]
        S_next = np.exp(log_S_next)
        
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next_clipped
        # V is defined on S_grid (log-spaced), so interpolate in log space
        log_S_grid = np.log(S_grid)
        log_S_next_clipped = np.log(S_next_clipped)
        
        # For each S_i, interpolate V at each S_next
        # Use np.interp which works on linearly spaced x, but our grid is log-spaced
        # So interpolate in log space: V is a function of log(S)
        
        # Create a 2D array of interpolated values
        V_interp = np.zeros_like(S_next_clipped)
        for i in range(n_grid):
            V_interp[i, :] = np.interp(log_S_next_clipped[i, :], log_S_grid, V)
        
        # Compute expected continuation value
        cont_vals = np.sum(V_interp * weights[np.newaxis, :], axis=1)
        cont_vals = disc * cont_vals
        
        # Optimal value: max(intrinsic, continuation)
        intrinsic = np.maximum(K - S_grid, 0.0)
        V_opt = np.maximum(intrinsic, cont_vals)
        
        # Find the boundary: the largest S where intrinsic > continuation
        # i.e., where it's optimal to exercise
        exercise_mask = intrinsic >= cont_vals
        # Find the largest S in exercise_mask
        if np.any(exercise_mask):
            # Get indices where exercise is optimal
            ex_indices = np.where(exercise_mask)[0]
            max_idx = ex_indices[-1]
            boundaries[j] = S_grid[max_idx]
        else:
            boundaries[j] = 0.0
        
        # Update V for next iteration
        V = V_opt
    
    # Ensure boundaries are non-increasing as j decreases (earlier dates have lower or equal boundaries)
    # Actually, for puts, earlier exercise boundaries are typically higher than later ones
    # because the time value of waiting decreases... let me think.
    # Actually for a put, the optimal exercise boundary increases as we approach maturity.
    # At the last date, it's K. Earlier, it's less than K because waiting has value.
    # So boundaries[0] <= boundaries[1] <= ... <= boundaries[n-1] = K
    # Let's enforce monotonicity
    for j in range(1, n_exercises):
        if boundaries[j] < boundaries[j-1]:
            boundaries[j] = boundaries[j-1]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices for all paths at this time step.
            time: the time step index (0-based, 0 to 11 for 12 dates)
            """
            # Ensure time is within bounds
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            # Exercise if S < B (strictly in the money relative to boundary)
            # If S == B, either choice is optimal; use <= to be consistent
            return current_state <= B
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
