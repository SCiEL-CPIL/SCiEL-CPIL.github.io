"""Bermudan put exercise policy via Longstaff-Schwartz Monte Carlo regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # Get training paths shape
    # training_paths: (n_paths, n_time_steps) or similar
    if isinstance(training_paths, np.ndarray):
        paths = training_paths
    else:
        paths = np.array(training_paths)
    
    # Determine number of paths and time steps
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    n_paths, n_steps = paths.shape
    
    # Ensure we have enough time steps
    # We need n_exercise exercise dates after time 0
    # paths should have n_exercise + 1 columns (time 0 to time n_exercise)
    if n_steps < n_exercise + 1:
        # If not enough steps, use what we have
        n_exercise = min(n_exercise, n_steps - 1)
        dt = T / (n_exercise + 1) if n_exercise > 0 else T
        exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # Extract price paths at exercise dates
    # paths[:, t] gives price at time t*dt
    # Exercise dates are at indices 1, 2, ..., n_exercise
    exercise_prices = paths[:, 1:n_exercise + 1]  # shape (n_paths, n_exercise)
    
    # Payoff for put: max(K - S, 0)
    terminal_payoff = np.maximum(K - exercise_prices[:, -1], 0.0)
    
    # Discount factors
    disc = np.exp(-r * dt)
    
    # Continuation values (work backwards)
    continuation = np.zeros((n_paths, n_exercise))
    exercise_value = np.maximum(K - exercise_prices, 0.0)
    
    # Start from terminal
    continuation[:, -1] = terminal_payoff
    
    # Regression basis functions
    def make_basis(S, t_idx, n_ex):
        """Create polynomial basis features for regression at time t_idx."""
        # Use normalized spot and time features
        S_norm = S / K  # normalize by strike
        t_norm = (n_ex - t_idx) / n_ex  # time to maturity normalized
        
        # Polynomial basis up to degree 2
        features = np.column_stack([
            np.ones_like(S_norm),
            S_norm,
            S_norm ** 2,
            t_norm * np.ones_like(S_norm),
            S_norm * t_norm * np.ones_like(S_norm),
            S_norm ** 2 * t_norm * np.ones_like(S_norm),
        ])
        return features
    
    # Learn thresholds via regression
    # For each exercise date from last to first
    thresholds = np.zeros(n_exercise)
    learned = False
    
    for t_idx in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 10.0:
            break
        
        # In-the-money mask
        itm_mask = exercise_prices[:, t_idx] < K
        
        if np.sum(itm_mask) < 10:
            # Not enough ITM samples, use simple threshold
            thresholds[t_idx] = K
            continue
        
        # Target: continuation value
        target = continuation[:, t_idx + 1] * disc
        
        # Features for ITM paths
        S_itm = exercise_prices[itm_mask, t_idx]
        target_itm = target[itm_mask]
        
        # Build basis
        features = make_basis(S_itm, t_idx, n_exercise)
        
        # Solve least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(features, target_itm, rcond=None)
            # Predict continuation for all ITM paths
            pred_cont = features @ coeffs
        except:
            pred_cont = target_itm
        
        # Threshold: exercise if intrinsic > continuation
        # Find the price level where intrinsic = continuation
        # For put: K - S = cont => S = K - cont
        # Use median of (K - pred_cont) as threshold
        threshold = np.median(K - pred_cont)
        thresholds[t_idx] = threshold
        
        # Update continuation: max(intrinsic, discounted continuation)
        intrinsic = np.maximum(K - exercise_prices[:, t_idx], 0.0)
        continuation[:, t_idx] = np.maximum(intrinsic, target)
    
    # If we didn't learn properly, use simple intrinsic-only policy
    if not learned and time.time() > deadline - 5.0:
        thresholds = K * np.ones(n_exercise)
    
    # Create exercise policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_exercise = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            current_state: array of current spot prices
            time: current time (or index)
            Returns: boolean array, True if should exercise
            """
            if isinstance(current_state, (list, tuple)):
                current_state = np.array(current_state)
            else:
                current_state = np.asarray(current_state)
            
            # Determine which exercise date we're at
            if isinstance(time, (int, np.integer)):
                t_idx = int(time)
            else:
                # time is a float, find closest exercise date
                t_idx = int(round(time / dt))
            
            # Clamp index
            t_idx = max(0, min(t_idx, self.n_exercise - 1))
            
            threshold = self.thresholds[t_idx]
            
            # Exercise if S < threshold (for put)
            return current_state < threshold
    
    policy = ExercisePolicy(thresholds, K)
    return policy
# EVOLVE-BLOCK-END
