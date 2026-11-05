"""Bermudan put exercise policy via Least Squares Monte Carlo (LSMC)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using LSMC.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_dates+1) or similar path data
    - contract: dict with K, r, sigma, T, n_exercise, etc.
    
    Returns an object with .exercise(current_state, time) method.
    """
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    dt = T / n_exercise
    
    # Training paths: assume shape (n_paths, n_exercise+1) with spot prices
    # or (n_paths, n_exercise) if no time-0
    paths = np.asarray(training_paths, dtype=np.float64)
    
    # Ensure 2D array
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    n_paths = paths.shape[0]
    
    # Determine number of dates from paths
    # If paths has n_exercise+1 columns, first is time 0
    # We exercise at dates 1..n_exercise (no time-0 exercise)
    
    # Payoff for put: max(K - S, 0)
    # Discount factor from date j to j-1: exp(-r*dt)
    disc = np.exp(-r * dt)
    
    # We'll work backward from the last exercise date
    # For each date j (from n_exercise down to 1), we estimate continuation value
    
    # Basis functions for regression
    # Use polynomial features in log(S) or S
    def make_basis(S):
        """Create feature matrix for regression. S is 1D array of spot prices."""
        S = np.asarray(S, dtype=np.float64)
        logS = np.log(np.maximum(S, 1e-10))
        # Features: 1, S, S^2, logS, logS^2, S*logS, (S-K), max(S-K,0)
        X = np.column_stack([
            np.ones_like(S),
            S,
            S**2,
            logS,
            logS**2,
            S * logS,
            S - K,
            np.maximum(S - K, 0.0)
        ])
        return X
    
    # We need to simulate or use training paths
    # training_paths should give us spot prices at each exercise date
    
    # Let's assume paths has shape (n_paths, n_dates) where n_dates = n_exercise
    # and paths[:, j] is the spot at exercise date j+1 (1-indexed)
    
    # If paths has n_exercise+1 columns, drop the first (time 0)
    if paths.shape[1] == n_exercise + 1:
        paths = paths[:, 1:]  # Now shape (n_paths, n_exercise)
    
    n_dates = paths.shape[1]
    if n_dates < n_exercise:
        n_dates = paths.shape[1]
    else:
        n_dates = n_exercise
    
    # Backward induction using LSMC
    # V[n_dates-1] = payoff at last date (no continuation)
    # V[j] = max(payoff_j, disc * E[V[j+1] | S_j])
    
    # Store exercise boundaries or continuation values
    # We'll store the regression coefficients for each date
    
    n_basis = 8
    coeffs = [None] * n_dates  # coeffs[j] for date j
    
    # Last date: no continuation, just exercise if in the money
    # Actually at the last exercise date, you always exercise if in the money
    # because there's no future option value
    
    # Start from the second-to-last date and go backward
    # V at last date = payoff
    # For date j (0-indexed in paths, corresponding to exercise date j+1):
    # If j == n_dates-1: V = payoff (terminal)
    # If j < n_dates-1: V = max(payoff, disc * E[V_{j+1} | S_j])
    
    # Let's work with 0-indexed dates in the paths array
    # paths[:, j] is spot at exercise date j+1
    
    # Initialize continuation value at last date
    # At the last exercise date, continuation is 0 (no more dates)
    # So V_last = payoff_last
    
    # We'll compute V at each date and store the regression for continuation
    
    # For each date from second-to-last to first:
    # 1. We have V_next (continuation value at next date) for each path
    # 2. Regress V_next on S_j to get E[V_next | S_j]
    # 3. Continuation value at j = disc * E[V_next | S_j]
    # 4. Exercise if payoff_j > continuation_j
    
    # Store the fitted regression for each date so we can use it at scoring time
    
    # V_next starts as payoff at the last date
    S_last = paths[:, n_dates - 1]
    V_next = np.maximum(K - S_last, 0.0)
    
    # We need to store the regression coefficients for each date
    # coeffs[j] will be the coefficients for predicting V_{j+1} from S_j
    # Actually, let's store the continuation value function parameters
    
    # Let's store: for each date j, the coefficients that map S_j to continuation value
    # cont_coeffs[j] = coefficients such that cont_value = X(S_j) @ cont_coeffs[j]
    
    cont_coeffs = [None] * n_dates
    
    for j in range(n_dates - 2, -1, -1):
        S_j = paths[:, j]
        
        # Regress V_next on S_j
        X_j = make_basis(S_j)
        
        # Solve least squares
        try:
            coeff, residuals, rank, sv = np.linalg.lstsq(X_j, V_next, rcond=None)
        except:
            # Fallback: simple average
            coeff = np.zeros(n_basis)
            coeff[0] = np.mean(V_next)
        
        # Continuation value at date j = disc * E[V_next | S_j]
        cont_val_j = disc * (X_j @ coeff)
        cont_val_j = np.maximum(cont_val_j, 0.0)  # Continuation value should be non-negative
        
        # Store coefficients for this date
        cont_coeffs[j] = coeff
        
        # Payoff at date j
        payoff_j = np.maximum(K - S_j, 0.0)
        
        # V_j = max(payoff_j, cont_val_j)
        V_next = np.maximum(payoff_j, cont_val_j)
        
        # Check deadline
        if time.time() > deadline - 10.0:
            break
    
    # Now we have cont_coeffs for each date
    # At scoring time, for a given spot S and date j, we compute:
    # - payoff = max(K - S, 0)
    # - continuation = X(S) @ cont_coeffs[j] (if j < n_dates-1)
    # - exercise if payoff > continuation
    
    # For the last date, always exercise if in the money
    # cont_coeffs[n_dates-1] is None, meaning continuation = 0
    
    class ExercisePolicy:
        def __init__(self, cont_coeffs, n_dates, K, r, dt, disc):
            self.cont_coeffs = cont_coeffs
            self.n_dates = n_dates
            self.K = K
            self.r = r
            self.dt = dt
            self.disc = disc
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            - current_state: array of spot prices (1D)
            - time: the time index or actual time
            
            Returns:
            - boolean array indicating whether to exercise
            """
            S = np.asarray(current_state, dtype=np.float64)
            
            # Determine which exercise date this corresponds to
            # time could be the index (0-indexed) or actual time
            if isinstance(time, (int, np.integer)):
                j = int(time)
            else:
                # Assume time is actual time, convert to index
                j = int(round(time / self.dt))
            
            # Clamp j to valid range
            j = max(0, min(j, self.n_dates - 1))
            
            # Compute payoff
            payoff = np.maximum(self.K - S, 0.0)
            
            # Compute continuation value
            if j < self.n_dates - 1 and self.cont_coeffs[j] is not None:
                X = make_basis(S)
                cont_val = self.disc * (X @ self.cont_coeffs[j])
                cont_val = np.maximum(cont_val, 0.0)
            else:
                cont_val = np.zeros_like(S)
            
            # Exercise if payoff > continuation
            return payoff > cont_val
    
    return ExercisePolicy(cont_coeffs, n_dates, K, r, dt, disc)
# EVOLVE-BLOCK-END
