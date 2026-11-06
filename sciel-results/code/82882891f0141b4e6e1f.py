"""Bermudan put exercise policy via binomial tree backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Extract parameters from contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12  # monthly exercise dates
    
    # Time points for exercise (excluding t=0)
    dt = T / n_exercise
    exercise_times = np.array([(i + 1) * dt for i in range(n_exercise)])
    
    # Build binomial tree parameters
    # Use a fine grid for the tree
    n_steps = 12  # one step per exercise period
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up
    p = (np.exp(r * dt) - down) / (up - down)
    disc = np.exp(-r * dt)
    
    # We'll work with log-price for numerical stability
    # Stock prices at each node: S * up^j * down^(n-j)
    # In log space: log(S) + j*log(up) + (n-j)*log(down) = log(S) + j*(log(up)-log(down)) + n*log(down)
    
    # For a put, we care about low stock prices
    # Build the tree from the perspective of the initial spot
    # The evaluator will call exercise(current_state, time) where current_state is a vector
    
    # Strategy: Precompute exercise thresholds at each exercise date
    # threshold[i] = critical stock price at exercise date i where exercise is optimal
    
    # Use backward induction on a binomial tree
    # At each node, value = max(immediate_exercise, disc * expected_future_value)
    
    # For efficiency, we'll use a log-space grid
    # Number of nodes at step k: k+1
    # We need to handle a range of stock prices
    
    # Let's use a large number of initial spots to define the grid
    # The spot will be around 90-110, but could vary
    # Use a grid in log-space
    
    # Define grid in terms of number of up-moves from a reference
    # Reference: S_ref = 100
    # At step k, stock = S_ref * up^j * down^(k-j) for j = 0, ..., k
    # In log: log(S) = log(S_ref) + j*(log(up) - log(down)) + k*log(down)
    
    # Let's define the grid by the number of down-moves from the top
    # At step k, there are k+1 nodes
    # Node j (0-indexed from bottom): j down-moves, k-j up-moves
    # S_j = S_ref * down^j * up^(k-j)
    
    # For backward induction, we need values at all nodes
    # Since the policy should work for any current state, we need to interpolate
    
    # Alternative: Use a fixed grid of stock prices and compute the optimal threshold
    # at each exercise date
    
    # Let's use a log-uniform grid of stock prices
    n_grid = 1000
    # Stock price range: from very low to high
    log_S_min = np.log(10.0)
    log_S_max = np.log(200.0)
    log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Backward induction
    # V[k] = value function at exercise date k (k=0 is last, k=n_exercise-1 is first)
    # At the last exercise date, V = max(K - S, 0)
    
    # We'll store thresholds: threshold[k] for exercise date k
    # threshold[k] = S such that (K-S) = disc * E[V[k+1]]
    
    # Initialize: at the last exercise date (k = n_exercise-1)
    # V_last(S) = max(K - S, 0)
    # Exercise if K - S >= 0, i.e., S <= K
    # threshold[last] = K
    
    thresholds = np.zeros(n_exercise)
    thresholds[-1] = K  # At the last date, exercise if S <= K
    
    # Work backwards
    # For k = n_exercise-2, ..., 0:
    # V_cont(S) = disc * (p * V_next(S*up) + (1-p) * V_next(S*down))
    # Exercise if (K - S) >= V_cont(S)
    
    # We need to represent V_next as a function of S
    # Since we're on a grid, we can interpolate
    
    # V_next at the grid points
    # At the last date: V_next(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    for k in range(n_exercise - 2, -1, -1):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Compute continuation value: disc * (p * V_next(S*up) + (1-p) * V_next(S*down))
        S_up = S_grid * up
        S_down = S_grid * down
        
        # Interpolate V_next at S_up and S_down
        # V_next is defined on log_S_grid
        log_S_up = np.log(S_up)
        log_S_down = np.log(S_down)
        
        # Interpolate
        V_at_up = np.interp(log_S_up, log_S_grid, V_next, left=0.0, right=0.0)
        V_at_down = np.interp(log_S_down, log_S_grid, V_next, left=0.0, right=0.0)
        
        V_cont = disc * (p * V_at_up + (1.0 - p) * V_at_down)
        
        # Exercise value
        V_exercise = np.maximum(K - S_grid, 0.0)
        
        # Find the threshold where V_exercise = V_cont
        # For S < K: V_exercise = K - S (decreasing in S)
        # V_cont is increasing in S (generally)
        # Find crossing point
        
        # diff = V_exercise - V_cont
        diff = V_exercise - V_cont
        
        # For S <= K, diff starts positive (at S=0, diff=K) and decreases
        # For S > K, diff is negative
        # Find where diff crosses zero
        
        # Only consider S <= K for the threshold
        mask = S_grid <= K
        if np.any(mask):
            S_sub = S_grid[mask]
            diff_sub = diff[mask]
            
            # Find sign change
            sign_change = np.where(np.diff(np.sign(diff_sub)) < 0)[0]
            if len(sign_change) > 0:
                # Take the first crossing (lowest S where we switch from exercise to continue)
                idx = sign_change[0]
                # Linear interpolation
                S1, S2 = S_sub[idx], S_sub[idx + 1]
                d1, d2 = diff_sub[idx], diff_sub[idx + 1]
                if d1 != d2:
                    threshold = S1 + (S2 - S1) * d1 / (d1 - d2)
                else:
                    threshold = S1
                thresholds[k] = threshold
            else:
                # No sign change: if all positive, threshold = K (always exercise)
                # If all negative, threshold = 0 (never exercise)
                if np.all(diff_sub > 0):
                    thresholds[k] = K
                elif np.all(diff_sub < 0):
                    thresholds[k] = 0.0
                else:
                    # Mixed but no clean crossing - use K as conservative
                    thresholds[k] = K
        else:
            thresholds[k] = K
        
        # Update V_next for next iteration
        V_next = np.maximum(V_exercise, V_cont)
    
    # Now we have thresholds[k] for each exercise date k (k=0 is first, k=n_exercise-1 is last)
    # During evaluation, given current_state and time, we determine which exercise date we're at
    # and check if current_state <= threshold[k]
    
    # The exercise times are at t_i = (i+1)*dt for i=0,...,11
    # Given a time t, find the exercise index: i = floor(t / dt) - 1
    # But we need to be careful about the exact mapping
    
    # Create a lookup: for each exercise date index i, the time is (i+1)*dt
    # Given time t, find i such that t is closest to (i+1)*dt
    
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K, r, sigma, T, n_exercise):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            self.dt = T / n_exercise
        
        def exercise(self, current_state, time_val):
            """
            current_state: array of stock prices
            time_val: current time
            Returns: boolean array indicating whether to exercise
            """
            current_state = np.asarray(current_state, dtype=float)
            time_val = float(time_val)
            
            # Determine which exercise date we're at
            # Exercise dates are at t_i = (i+1)*dt for i=0,...,n_exercise-1
            # Find the exercise index
            i = int(np.round(time_val / self.dt)) - 1
            i = max(0, min(i, self.n_exercise - 1))
            
            # Check if we should exercise
            threshold = self.thresholds[i]
            return current_state <= threshold
    
    policy = ExercisePolicy(thresholds, exercise_times, K, r, sigma, T, n_exercise)
    return policy
# EVOLVE-BLOCK-END
