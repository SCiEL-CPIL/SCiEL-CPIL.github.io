"""
Bermudan Put Exercise Policy using Backward Induction with Gauss-Hermite Quadrature.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan put exercise policy.
    
    Args:
        training_paths: Not used directly for the core algorithm, as we use a 
                        deterministic numerical method. Included for API compliance.
        contract: Dictionary containing contract parameters.
        
    Returns:
        An object with an 'exercise' method.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Extract parameters from contract
    # Based on task description: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # Contract might contain these, but we default to the specified values if not present or if we want to be robust.
    # The task says "fixed model/data assumptions", so we can hardcode or read from contract.
    # Let's try to read from contract, falling back to defaults.
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_exercise_dates = contract.get('num_exercise_dates', 12)
    
    # Exercise dates are monthly, no time-0 exercise
    # t_i = i * T / num_exercise_dates for i in 1..num_exercise_dates
    dt = T / num_exercise_dates
    times = np.linspace(0, T, num_exercise_dates + 1)[1:] # t_1 to t_12

    # Setup Grid
    # We work in log-space for better distribution
    # Range of log(S): S can go from very low to high. 
    # For a put, low S is critical. 
    # Let's choose a range that covers 99% of the distribution at T.
    # log(S_T) ~ N(log(S_0) + (r - sigma^2/2)T, sigma*sqrt(T))
    # We need a grid that covers the potential range of S at all times.
    # Min S: S_0 * exp((r - sigma^2/2)*T - 4*sigma*sqrt(T))
    # Max S: S_0 * exp((r - sigma^2/2)*T + 4*sigma*sqrt(T))
    
    # Since the policy is state-dependent and we need to handle any current_state,
    # the grid should cover a reasonable range. 
    # However, the exercise decision is local. 
    # We compute the value function on a grid of log(S).
    
    # Gauss-Hermite Quadrature
    n_ghq = 60  # Number of quadrature points
    nodes, weights = np.polynomial.hermite.hermgauss(n_ghq)
    
    # GHQ integrates f(x) * exp(-x^2). 
    # We need to integrate g(Z) where Z ~ N(0,1).
    # Let Z = sqrt(2) * x. 
    # dZ = sqrt(2) dx.
    # E[g(Z)] = (1/sqrt(2*pi)) * integral g(sqrt(2)*x) * exp(-x^2) * sqrt(2) dx
    # = (1/sqrt(pi)) * integral g(sqrt(2)*x) * exp(-x^2) dx
    # So weights_ghq = weights * sqrt(2) / sqrt(pi)
    # And nodes_ghq = nodes * sqrt(2)
    
    sqrt_2 = np.sqrt(2.0)
    ghq_nodes = nodes * sqrt_2
    ghq_weights = weights * sqrt_2 / np.sqrt(np.pi)
    
    # Grid of log-spots
    # We need to define a range for log(S).
    # Let's center around log(100) = log(K).
    # Range: +/- 3 standard deviations of log(S) over T.
    # std_log = sigma * sqrt(T) = 0.2
    # log(S) range: log(100) +/- 0.2 * 3 = log(100) +/- 0.6
    # This might be too narrow for early times if S moves a lot, but the distribution narrows? No, variance increases with time.
    # At t=T, std is 0.2. At t=0, std is 0.
    # So the widest range is at T.
    # Let's use a wider range to be safe: +/- 1.0 in log space.
    log_min = np.log(K) - 1.5
    log_max = np.log(K) + 1.5
    n_grid = 400
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)
    
    # Backward Induction
    # V[i] is the value of the option at time t_i given the log-spot is log_grid
    # V[i] has shape (n_grid,)
    
    # Initialize V at the final exercise date (or T if we consider the terminal payoff)
    # The last exercise date is t_12 = T.
    # Payoff at T: (K - S_T)^+
    V_next = np.maximum(K - grid, 0.0)
    
    # Store thresholds for each exercise date
    # threshold[i] is the log-spot below which we exercise at time t_{i+1}
    # We store thresholds for t_1 to t_12
    thresholds = np.zeros(num_exercise_dates)
    
    # Iterate backwards from t_11 to t_1
    # Note: V_next corresponds to the value at t_{i+1}
    # We are computing V at t_i
    
    for i in range(num_exercise_dates - 1, -1, -1):
        # Current time t_i
        t_curr = times[i]
        t_next = times[i+1] if i+1 < num_exercise_dates else T
        
        # Discount factor
        disc = np.exp(-r * dt)
        
        # Compute Continuation Value
        # E[ V_next(S_{t+dt}) | S_t ]
        # S_{t+dt} = S_t * exp( (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z )
        # log(S_{t+dt}) = log(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
        # Let mu = (r - sigma^2/2)*dt
        # Let sd = sigma*sqrt(dt)
        
        mu = (r - 0.5 * sigma**2) * dt
        sd = sigma * np.sqrt(dt)
        
        # For each log_spot in log_grid, we compute the expected value
        # log_S_next = log_grid + mu + sd * Z
        # Z is sampled from GHQ nodes
        
        # Vectorized operation:
        # log_grid shape: (n_grid,)
        # ghq_nodes shape: (n_ghq,)
        # We want a matrix of shape (n_grid, n_ghq)
        # log_S_next = log_grid[:, np.newaxis] + mu + sd * ghq_nodes[np.newaxis, :]
        
        log_S_next = log_grid[:, np.newaxis] + mu + sd * ghq_nodes[np.newaxis, :]
        
        # We need to interpolate V_next at log_S_next
        # V_next is defined on log_grid.
        # np.interp requires increasing x. log_grid is increasing.
        # log_S_next might be outside [log_min, log_max].
        # We can clip log_S_next to the grid range, or extend the grid.
        # Clipping is a reasonable approximation for extreme tails, as probabilities are low.
        log_S_next_clipped = np.clip(log_S_next, log_min, log_max)
        
        # Interpolate
        # V_next_interp shape: (n_grid, n_ghq)
        V_next_interp = np.interp(log_S_next_clipped, log_grid, V_next)
        
        # Weighted sum
        # Continuation Value = sum( V_next_interp * ghq_weights )
        # ghq_weights shape: (n_ghq,)
        cont_val = np.sum(V_next_interp * ghq_weights[np.newaxis, :], axis=1)
        
        # Discounted Continuation Value
        disc_cont_val = disc * cont_val
        
        # Immediate Exercise Payoff
        # Payoff = (K - S_t)^+ = (K - exp(log_grid))^+
        exercise_payoff = np.maximum(K - grid, 0.0)
        
        # Optimal Value
        V_curr = np.maximum(exercise_payoff, disc_cont_val)
        
        # Determine Exercise Threshold
        # We exercise if exercise_payoff > disc_cont_val
        # Find the largest log_grid where we exercise
        # Since payoff decreases with S and cont_val generally increases (for a put? No, cont_val also decreases with S, but at different rates), 
        # the region where we exercise is S < Threshold.
        # Let's find the crossover.
        # diff = exercise_payoff - disc_cont_val
        # We exercise where diff > 0.
        diff = exercise_payoff - disc_cont_val
        
        # Find the index where diff changes from positive to negative
        # diff is decreasing in S (log_grid)
        # Find the last index where diff > 0
        positive_mask = diff > 0
        if np.any(positive_mask):
            # Find the boundary
            # If all are positive, threshold is log_max
            # If none are positive, threshold is log_min (or -inf, but we exercise everywhere? No, if none positive, we never exercise? 
            # Wait, if diff < 0 everywhere, we never exercise. 
            # But for a put, at very low S, payoff is high and cont_val is low (near 0). So diff should be positive at low S.
            # So there should be some positive values.
            
            # Find the transition point
            # We want the largest log_grid such that diff > 0.
            # Since diff is monotonic decreasing, we can use searchsorted or simple logic.
            # Let's find the index of the last positive value.
            last_pos_idx = np.where(positive_mask)[0][-1]
            
            # Interpolate to find the exact threshold
            if last_pos_idx < n_grid - 1:
                # Linear interpolation between last_pos_idx and last_pos_idx + 1
                x0 = log_grid[last_pos_idx]
                x1 = log_grid[last_pos_idx + 1]
                y0 = diff[last_pos_idx]
                y1 = diff[last_pos_idx + 1]
                
                if y1 - y0 != 0:
                    # Root of y0 + (x - x0) * (y1 - y0)/(x1 - x0) = 0
                    # x = x0 - y0 * (x1 - x0) / (y1 - y0)
                    threshold = x0 - y0 * (x1 - x0) / (y1 - y0)
                else:
                    threshold = x0
            else:
                threshold = log_max
        else:
            # No exercise region? This shouldn't happen for a put with reasonable parameters.
            # But if it does, set threshold to log_min (exercise nowhere? or everywhere?)
            # If diff < 0 everywhere, we continue everywhere.
            threshold = log_min - 1.0 # Effectively -inf in log space
        
        thresholds[i] = threshold
        
        V_next = V_curr
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Define the Policy Object
    class ExercisePolicy:
        def __init__(self, thresholds, log_min, log_max):
            self.thresholds = thresholds
            self.log_min = log_min
            self.log_max = log_max
            
        def exercise(self, current_state, time_index):
            """
            current_state: numpy array of spot prices
            time_index: int, index of the current exercise date (0-based, corresponding to t_1...t_12)
            Returns: boolean array
            """
            # time_index is 0-based in the contract's 12 dates?
            # The task says "12 monthly exercise dates".
            # The `time` argument in `exercise(current_state, time)` is likely the time value or the index.
            # "returning a boolean vector"
            # "candidate receives current states only during scoring"
            # Usually, `time` is the current time value (float) or the step index.
            # Given the signature `exercise(current_state, time)`, and the context of discrete steps,
            # it's safer to assume `time` is the index of the exercise date (0 to 11) or the time value.
            # Let's check the baseline or common patterns. 
            # If `time` is a float, we map it to the nearest exercise date.
            # If `time` is an int, we use it directly.
            
            # Let's assume `time` is the index (0..11) as is common in such loops, 
            # OR it is the time value. 
            # The prompt says "exercise(current_state, time)". 
            # In many similar tasks, `time` is the step index.
            # Let's handle both: if time is float, find nearest index. If int, use as index.
            
            if isinstance(time, float) or isinstance(time, np.float64):
                # Map time value to index
                # t_i = (i+1) * dt
                # i = round(t / dt) - 1
                idx = int(np.round(time / dt) - 1)
                idx = np.clip(idx, 0, num_exercise_dates - 1)
            else:
                idx = int(time)
                idx = np.clip(idx, 0, num_exercise_dates - 1)
            
            threshold = self.thresholds[idx]
            
            # Exercise if log(S) < threshold
            # current_state is spot prices
            log_current_state = np.log(current_state)
            
            return log_current_state < threshold

    return ExercisePolicy(thresholds, log_min, log_max)

# EVOLVE-BLOCK-END
