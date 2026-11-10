"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = 150.0 - 10.0  # 140s safety margin
    
    # Extract parameters from contract
    # Assume contract has structure with spot, K, r, sigma, T, n_exercises, n_paths, etc.
    # Based on the task description: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # 12 monthly exercise dates, 4096 training paths, 8192 evaluation paths
    
    try:
        # Try to extract parameters from contract
        if hasattr(contract, 'params'):
            params = contract.params
        elif isinstance(contract, dict):
            params = contract
        else:
            # Default parameters from task description
            params = {}
        
        S0_list = params.get('S0', [90, 100, 110])
        K = params.get('K', 100.0)
        r = params.get('r', 0.05)
        sigma = params.get('sigma', 0.2)
        T = params.get('T', 1.0)
        n_exercises = params.get('n_exercises', 12)
        
        # Get training paths
        # training_paths should be an array of shape (n_paths, n_exercises+1) or similar
        # Let's assume it's (n_paths, n_timesteps) where n_timesteps includes time 0
        paths = np.asarray(training_paths)
        
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths = paths.shape[0]
        n_timesteps = paths.shape[1]
        
        # Exercise dates are monthly, 12 dates, no time-0 exercise
        # So exercise happens at timesteps 1, 2, ..., 12 (if 12 monthly steps)
        # T = 1 year, 12 monthly steps, dt = 1/12
        
        dt = T / n_exercises
        
        # Backward induction for Bermudan put
        # At each exercise date, we compare immediate payoff vs continuation value
        # For a put: payoff = max(K - S, 0)
        
        # We'll use regression-based approach (Longstaff-Schwarz style)
        # Or simpler: compute continuation values via discounting expected future values
        
        # Simple approach: backward induction with Monte Carlo averaging
        # But we need to be careful - the candidate only sees current states during scoring
        # So we need to fit a policy that generalizes
        
        # Strategy: For each exercise date, fit a threshold or regression
        # For a put, the exercise region is typically S < boundary(t)
        
        # Let's use a parametric approach: fit a boundary for each exercise date
        # Using the training data to determine optimal exercise decisions
        
        # First, compute the optimal policy via backward induction on training paths
        # Then fit a simple model to generalize
        
        # Backward induction:
        # V_n(S) = max(K - S, exp(-r*dt) * E[V_{n+1}(S_{n+1}) | S_n = S])
        
        # Since we have paths, we can approximate the expectation by averaging over paths
        # that are at or near each state
        
        # For simplicity, let's use a grid-based approach
        # Create a grid of stock prices
        s_min = max(1.0, paths.min() * 0.9)
        s_max = paths.max() * 1.1
        n_grid = 200
        grid = np.linspace(s_min, s_max, n_grid)
        
        # Initialize continuation value at final date
        # At the last exercise date, continuation value is 0 (no more exercise opportunities)
        # Actually, for Bermudan, after the last exercise date, the option expires
        
        # Let's define V[j] as the value at exercise date j (j = 0 to n_exercises-1)
        # V[n_exercises] = 0 (after all exercise dates)
        
        # Work backward from the last exercise date
        # At each date, for each grid point, compute:
        #   immediate = max(K - S, 0)
        #   continuation = exp(-r*dt) * E[V_next(S_next) | S_current]
        #   V_current = max(immediate, continuation)
        #   exercise if immediate > continuation
        
        # To compute E[V_next(S_next) | S_current], we need the transition distribution
        # Under risk-neutral GBM: S_next = S_current * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        
        # We can use quadrature or Monte Carlo to approximate this expectation
        
        # Let's use a simple approach: for each grid point, find paths that are near
        # that grid point at the current time, and average their next-step values
        
        # This is inefficient but should work for 200 grid points and 12 steps
        
        # Alternative: use the analytic transition to compute the expectation
        # For GBM, given S, the distribution of S_next is lognormal
        # We can use Gauss-Hermite quadrature or just sample
        
        # Let's use a parametric regression approach instead
        # Fit a linear model: V_next(S_next) ≈ a + b * S_next + c * S_next^2
        # Then E[V_next(S_next) | S] can be computed analytically for lognormal
        
        # This is more efficient and accurate
        
        # Let's implement this
        
        # Store boundaries for each exercise date
        boundaries = []
        
        # Work backward
        # V_next is a function of S, approximated by a quadratic
        # Start with V_next(S) = 0 at the final date
        
        v_next_coeffs = np.array([0.0, 0.0, 0.0])  # a, b, c for a + b*S + c*S^2
        
        for j in range(n_exercises - 1, -1, -1):
            # Compute continuation value for each grid point
            # S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            # E[V_next(S_next) | S] = E[a + b*S_next + c*S_next^2]
            # = a + b*E[S_next] + c*E[S_next^2]
            
            # For GBM:
            # E[S_next | S] = S * exp(r*dt)
            # E[S_next^2 | S] = S^2 * exp(2*r*dt + sigma^2*dt)
            
            drift = (r - 0.5 * sigma**2) * dt
            vol = sigma * np.sqrt(dt)
            
            # E[S_next] = S * exp(r*dt)
            # E[S_next^2] = S^2 * exp(2*r*dt + sigma^2*dt)
            
            exp_r_dt = np.exp(r * dt)
            exp_2r_sigma2_dt = np.exp(2 * r * dt + sigma**2 * dt)
            
            # For each grid point S:
            # E[S_next] = S * exp_r_dt
            # E[S_next^2] = S^2 * exp_2r_sigma2_dt
            
            # V_cont(S) = exp(-r*dt) * (a + b*S*exp_r_dt + c*S^2*exp_2r_sigma2_dt)
            
            a, b, c = v_next_coeffs
            
            v_cont = np.exp(-r * dt) * (a + b * grid * exp_r_dt + c * grid**2 * exp_2r_sigma2_dt)
            
            # Immediate payoff
            v_immediate = np.maximum(K - grid, 0.0)
            
            # Value at this date
            v_current = np.maximum(v_immediate, v_cont)
            
            # Exercise region: where immediate > continuation
            exercise_mask = v_immediate > v_cont
            
            # Find the boundary: the largest S where we exercise
            # For a put, exercise is optimal for S < boundary
            # So the boundary is the transition point
            
            if np.any(exercise_mask):
                # Find the last index where we exercise
                exercise_indices = np.where(exercise_mask)[0]
                # The boundary is between the last exercise point and the first non-exercise point
                if len(exercise_indices) > 0 and exercise_indices[-1] < n_grid - 1:
                    boundary = grid[exercise_indices[-1]]
                else:
                    boundary = grid[-1]
            else:
                boundary = grid[0]
            
            boundaries.append(boundary)
            
            # Now fit V_current to a quadratic for the next iteration
            # Use least squares
            # V_current ≈ a_new + b_new*S + c_new*S^2
            
            A = np.column_stack([np.ones(n_grid), grid, grid**2])
            try:
                v_next_coeffs, _, _, _ = np.linalg.lstsq(A, v_current, rcond=None)
            except:
                v_next_coeffs = np.array([0.0, 0.0, 0.0])
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        # boundaries is in reverse order (last date first), so reverse it
        boundaries = boundaries[::-1]
        
        # Pad if needed
        while len(boundaries) < n_exercises:
            boundaries.insert(0, K)  # Default: exercise if S < K
        
        # Create the policy object
        class ExercisePolicy:
            def __init__(self, boundaries, n_exercises):
                self.boundaries = boundaries
                self.n_exercises = n_exercises
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=float)
                time = np.asarray(time)
                
                if np.ndim(time) == 0:
                    time = np.full_like(current_state, time, dtype=float)
                
                n = len(current_state)
                result = np.zeros(n, dtype=bool)
                
                # Determine which exercise date each state corresponds to
                # time is in [0, T], exercise dates are at dt, 2dt, ..., T
                # Exercise date index = round(time/dt) - 1 (0-indexed)
                
                for i in range(n):
                    t = time[i]
                    s = current_state[i]
                    
                    # Find the exercise date index
                    # Exercise dates are at j*dt for j = 1, ..., n_exercises
                    # So date index = round(t/dt) - 1
                    date_idx = int(round(t / dt)) - 1
                    
                    if date_idx < 0:
                        date_idx = 0
                    elif date_idx >= self.n_exercises:
                        date_idx = self.n_exercises - 1
                    
                    # Exercise if S < boundary
                    if s < self.boundaries[date_idx]:
                        result[i] = True
                
                return result
        
        return ExercisePolicy(boundaries, n_exercises)
        
    except Exception as e:
        # Fallback: simple exercise if S < K
        class ExercisePolicyFallback:
            def __init__(self, K):
                self.K = K
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=float)
                return current_state < self.K
        
        K = params.get('K', 100.0) if 'params' in locals() else 100.0
        return ExercisePolicyFallback(K)

# EVOLVE-BLOCK-END
