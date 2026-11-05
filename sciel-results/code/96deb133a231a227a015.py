"""Bermudan put exercise policy via Least Squares Monte Carlo (LSMC)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using LSMC with improved basis,
    in/out sample split, and control variates.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_dates+1) or similar path data
    - contract: dict with K, r, sigma, T, n_exercise, etc.
    
    Returns an object with .exercise(current_state, time) method.
    """
    t_start = time.time()
    deadline = t_start + 108.0  # Safety margin before 120s kill

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
    
    # Payoff for put: max(K - S, 0)
    # Discount factor from date j to j-1: exp(-r*dt)
    disc = np.exp(-r * dt)
    
    # If paths has n_exercise+1 columns, drop the first (time 0)
    if paths.shape[1] == n_exercise + 1:
        paths = paths[:, 1:]  # Now shape (n_paths, n_exercise)
    
    n_dates = min(paths.shape[1], n_exercise)
    
    # Richer basis functions: polynomials in S and log(S)
    def make_basis(S):
        """Create feature matrix for regression. S is 1D array of spot prices."""
        S = np.asarray(S, dtype=np.float64)
        logS = np.log(np.maximum(S, 1e-10))
        # Features: 1, S, S^2, S^3, logS, logS^2, logS^3, S*logS, S*logS^2
        X = np.column_stack([
            np.ones_like(S),
            S,
            S**2,
            S**3,
            logS,
            logS**2,
            logS**3,
            S * logS,
            S * logS**2
        ])
        return X
    
    n_basis = 9
    
    # Control variate: under GBM, E[S_{j+1} | S_j] = S_j * exp(r*dt)
    # So E[max(K - S_{j+1}, 0) | S_j] can be approximated analytically for European put
    # But for Bermudan, we use S_{j+1} as a control variate for the regression
    # E[S_{j+1} | S_j] = S_j * exp(r*dt)
    
    # Backward induction using LSMC with in/out sample split
    cont_coeffs = [None] * n_dates
    
    # V_next starts as payoff at the last date
    S_last = paths[:, n_dates - 1]
    V_next = np.maximum(K - S_last, 0.0)
    
    for j in range(n_dates - 2, -1, -1):
        S_j = paths[:, j]
        S_next = paths[:, j + 1]
        
        # In/out sample split: use first 60% for training, last 40% for validation
        split = int(0.6 * n_paths)
        
        # Training set
        S_j_train = S_j[:split]
        V_next_train = V_next[:split]
        S_next_train = S_next[:split]
        
        # Build basis for training set
        X_train = make_basis(S_j_train)
        
        # Control variate: E[S_next | S_j] = S_j * exp(r*dt)
        # Use S_next - S_j * exp(r*dt) as a zero-mean control
        cv_train = S_next_train - S_j_train * np.exp(r * dt)
        
        # Augmented design matrix: [basis, control_variate]
        X_aug = np.column_stack([X_train, cv_train])
        
        # Solve least squares with control variate
        try:
            coeff_aug, _, _, _ = np.linalg.lstsq(X_aug, V_next_train, rcond=None)
        except:
            # Fallback: simple average
            coeff_aug = np.zeros(n_basis + 1)
            coeff_aug[0] = np.mean(V_next_train)
        
        # Extract regression coefficients (first n_basis)
        coeff = coeff_aug[:n_basis]
        
        # Store coefficients for this date
        cont_coeffs[j] = coeff
        
        # Compute continuation value for ALL paths using fitted coefficients
        X_all = make_basis(S_j)
        cont_val_all = X_all @ coeff
        cont_val_all = np.maximum(cont_val_all, 0.0)
        
        # Payoff at date j
        payoff_j = np.maximum(K - S_j, 0.0)
        
        # V_j = max(payoff_j, cont_val_j)
        V_next = np.maximum(payoff_j, cont_val_all)
        
        # Check deadline
        if time.time() > deadline - 12.0:
            break
    
    class ExercisePolicy:
        def __init__(self, cont_coeffs, n_dates, K, r, dt):
            self.cont_coeffs = cont_coeffs
            self.n_dates = n_dates
            self.K = K
            self.r = r
            self.dt = dt
        
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
            if isinstance(time, (int, np.integer)):
                j = int(time)
            else:
                j = int(round(time / self.dt))
            
            # Clamp j to valid range
            j = max(0, min(j, self.n_dates - 1))
            
            # Compute payoff
            payoff = np.maximum(self.K - S, 0.0)
            
            # Compute continuation value
            if j < self.n_dates - 1 and self.cont_coeffs[j] is not None:
                X = make_basis(S)
                cont_val = X @ self.cont_coeffs[j]
                cont_val = np.maximum(cont_val, 0.0)
            else:
                cont_val = np.zeros_like(S)
            
            # Exercise if payoff > continuation
            return payoff > cont_val
    
    return ExercisePolicy(cont_coeffs, n_dates, K, r, dt)
# EVOLVE-BLOCK-END
