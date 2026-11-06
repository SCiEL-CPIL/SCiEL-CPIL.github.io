"""Bermudan put exercise policy using Longstaff-Schwartz backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # training_paths shape: (n_paths, n_dates+1) or (n_paths, n_dates)
    # Let's infer from shape
    if training_paths.ndim == 2:
        n_paths, n_cols = training_paths.shape
    else:
        raise ValueError("training_paths must be 2D")
    
    # Assume training_paths contains prices at exercise dates
    # If n_cols == n_dates, prices at dates 1..n_dates (no time 0)
    # If n_cols == n_dates + 1, includes time 0
    if n_cols == n_dates:
        S = training_paths.copy()  # shape (n_paths, n_dates)
    elif n_cols == n_dates + 1:
        S = training_paths[:, 1:].copy()  # drop time 0
    else:
        # Fallback: use last n_dates columns
        S = training_paths[:, -n_dates:].copy()
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # We'll store the decision at each date: which paths exercise
    # and the continuation value estimates
    # Work backward from last date to first
    
    # V[t] = value at date t for each path (before decision)
    # For the last date (index n_dates-1), V = payoff
    V = np.zeros((n_paths, n_dates))
    
    # Start from the last exercise date
    V[:, n_dates - 1] = np.maximum(K - S[:, n_dates - 1], 0.0)
    
    # Exercise mask: True if path exercises at that date
    exercise_mask = np.zeros((n_paths, n_dates), dtype=bool)
    exercise_mask[:, n_dates - 1] = V[:, n_dates - 1] > 0  # always exercise if in-the-money at last date
    
    # For backward induction, we need to regress the continuation value
    # at date t on features of S[:, t] using paths that are still alive
    # A path is "alive" at date t if it has not exercised at any date > t
    
    # Let's track alive paths: alive[t] = True if path hasn't exercised at dates t+1..n_dates-1
    alive = np.ones(n_paths, dtype=bool)
    # At the last date, all paths are alive (they haven't exercised at any future date)
    
    # Actually, let me restructure: we go backward from t = n_dates-2 down to 0
    # At each step t, we have V[:, t+1] which is the value at date t+1
    # The continuation value at date t is: disc * E[V[:, t+1] | S[:, t]] for alive paths
    # We regress V[:, t+1] on features of S[:, t] using alive paths
    
    # Feature function: use polynomial features of S
    def make_features(s):
        """Create polynomial features for regression."""
        # Use degree 3 polynomial in log(S) and S itself
        log_s = np.log(s + 1e-10)
        s_norm = s / K  # normalize by strike
        features = np.column_stack([
            np.ones_like(s),
            s_norm,
            s_norm ** 2,
            s_norm ** 3,
            log_s,
            log_s ** 2
        ])
        return features
    
    # Store regression coefficients for each date so we can use them in exercise()
    reg_coeffs = []  # reg_coeffs[t] = coefficients for regression at date t
    
    # Go backward
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Paths alive at date t: those that haven't exercised at dates t+1, ..., n_dates-1
        # We track this by: a path is alive at t if exercise_mask[:, t+1:n_dates] are all False
        alive_at_t = np.all(~exercise_mask[:, t + 1:], axis=1)
        
        n_alive = np.sum(alive_at_t)
        if n_alive < 10:
            # Too few paths, just use immediate exercise
            V[:, t] = np.maximum(K - S[:, t], 0.0)
            exercise_mask[:, t] = V[:, t] > 0
            reg_coeffs.append(None)
            continue
        
        # Get the data for alive paths
        s_alive = S[alive_at_t, t]
        v_next_alive = V[alive_at_t, t + 1]
        
        # Create features
        X = make_features(s_alive)
        
        # Fit linear regression: v_next ≈ X @ coeffs
        # Use numpy lstsq for speed
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, v_next_alive, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X.shape[1])
        
        # Predict continuation value for all paths at date t
        X_all = make_features(S[:, t])
        cont_val = X_all @ coeffs  # shape (n_paths,)
        
        # Continuation value is discounted expected value
        cont_val = disc * cont_val
        
        # Immediate exercise payoff
        ex_val = np.maximum(K - S[:, t], 0.0)
        
        # Decision: exercise if ex_val > cont_val
        should_exercise = ex_val > cont_val
        
        # Value at date t
        V[:, t] = np.where(should_exercise, ex_val, cont_val)
        
        # Update exercise mask
        exercise_mask[:, t] = should_exercise
        
        # Store coefficients
        reg_coeffs.append(coeffs)
    
    # reg_coeffs is in reverse order (from t=n_dates-2 down to t=0)
    # Let's reverse it so reg_coeffs[t] corresponds to date t
    reg_coeffs = reg_coeffs[::-1]
    
    # Now create the policy object
    class BermudanExercisePolicy:
        def __init__(self, reg_coeffs, S0, K, r, sigma, T, n_dates):
            self.reg_coeffs = reg_coeffs
            self.S0 = S0
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_dates = n_dates
            self.dt = T / n_dates
            
        def _make_features(self, s):
            """Create polynomial features matching the training."""
            log_s = np.log(s + 1e-10)
            s_norm = s / self.K
            features = np.column_stack([
                np.ones_like(s),
                s_norm,
                s_norm ** 2,
                s_norm ** 3,
                log_s,
                log_s ** 2
            ])
            return features
        
        def exercise(self, current_state, time):
            """
            current_state: array of current spot prices
            time: current time index (0-based, corresponding to exercise date index)
            Returns: boolean vector, True if exercise
            """
            t = int(time)
            if t < 0 or t >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            # At the last date, always exercise if in-the-money
            if t == self.n_dates - 1:
                return current_state < self.K
            
            coeffs = self.reg_coeffs[t]
            if coeffs is None:
                # Fallback: exercise if in-the-money
                return current_state < self.K
            
            # Compute continuation value
            X = self._make_features(current_state)
            cont_val = np.exp(-self.r * self.dt) * (X @ coeffs)
            
            # Immediate exercise payoff
            ex_val = np.maximum(self.K - current_state, 0.0)
            
            return ex_val > cont_val
    
    policy = BermudanExercisePolicy(reg_coeffs, S0, K, r, sigma, T, n_dates)
    return policy
# EVOLVE-BLOCK-END
