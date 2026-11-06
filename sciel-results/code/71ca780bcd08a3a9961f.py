import numpy as np

def fit_exercise_policy(training_paths, contract):
    # ---- Extract model parameters from the contract (with safe defaults) ----
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    def _get(obj, *names, default=None):
        for n in names:
            if isinstance(obj, dict):
                if n in obj:
                    return obj[n]
            else:
                if hasattr(obj, n):
                    return getattr(obj, n)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'Vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', 'horizon', default=T))
        nd = _get(contract, 'n_dates', 'n_exercise', 'n_bermudan', 'exercise_dates', default=n_dates)
        if nd is not None:
            n_dates = int(nd)
    except Exception:
        pass

    # training_paths is expected to be a 2D array of shape (n_paths, n_dates)
    # where each row is a path and each column is the spot price at an exercise date.
    # The task says "12 monthly exercise dates (no time-0 exercise)".
    # So training_paths[:, 0] is S at t_1, ..., training_paths[:, 11] is S at t_12.
    
    S_paths = np.asarray(training_paths, dtype=float)
    n_paths, n_cols = S_paths.shape
    
    # Ensure we have the correct number of dates
    if n_cols < n_dates:
        # If not enough columns, pad or handle error? 
        # Assuming input is correct per contract.
        pass
        
    # Time grid for exercise dates
    # t_k = k * T / n_dates for k=1..n_dates
    # Index 0 in S_paths corresponds to t_1
    # Index n_dates-1 corresponds to t_n_dates
    
    # We will store regression coefficients for each date index 0..n_dates-1
    # coeffs[k] corresponds to the continuation value function at date k (t_{k+1})
    # Actually, let's map indices carefully.
    # Let date index j = 0..11 correspond to time t_{j+1}.
    # We iterate backwards from j=11 to 0.
    
    # Basis function degree
    deg = 4
    
    # Precompute basis functions for all paths at all dates?
    # Or compute on the fly.
    # For regression, we need X matrix for each step.
    
    coeffs = [None] * n_dates
    
    # Initialize values at the last date (maturity)
    # V at t_12 is Intrinsic
    V_next = np.maximum(K - S_paths[:, n_dates - 1], 0.0)
    
    # Iterate backwards
    # j goes from n_dates - 2 down to 0
    # At step j, we are at date t_{j+1}.
    # We want to fit the continuation value C_j(S) = E[Discount * V_{j+1} | S_j].
    # The target for regression is V_next (which is V at t_{j+2}).
    # Discount factor: dt = T / n_dates.
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    for j in range(n_dates - 2, -1, -1):
        # Target: Discounted V at next step
        target = disc * V_next
        
        # Features: Basis functions of S_paths[:, j]
        # Use log-spot for better conditioning
        log_S = np.log(S_paths[:, j] + 1e-10)
        
        # Construct design matrix X
        # Columns: 1, log_S, (log_S)^2, (log_S)^3, (log_S)^4
        X = np.ones((n_paths, deg + 1))
        for k in range(1, deg + 1):
            X[:, k] = log_S ** k
        
        # Solve least squares: X * c = target
        # Use lstsq for robustness
        c, residuals, rank, s = np.linalg.lstsq(X, target, rcond=None)
        coeffs[j] = c
        
        # Calculate Continuation Value C_j for all paths
        C_j = X @ c
        
        # Calculate Intrinsic Value I_j
        I_j = np.maximum(K - S_paths[:, j], 0.0)
        
        # Optimal Value V_j = max(I_j, C_j)
        # This V_j becomes V_next for the previous step (j-1)
        V_next = np.maximum(I_j, C_j)
        
    # Note: coeffs[n_dates-1] is not set because at maturity we always exercise if ITM?
    # Actually, at maturity, there is no continuation. The value is intrinsic.
    # The exercise policy at maturity is just Intrinsic > 0.
    # But our `exercise` function needs to handle all dates.
    # For the last date, we don't have a regression. We can set coeffs for the last date to None
    # and handle it specially, or just assume we never "continue" past maturity.
    # The exercise function will be called for times corresponding to the 12 dates.
    
    # Let's create a helper to evaluate the polynomial
    def _eval_poly(S, c):
        log_S = np.log(S + 1e-10)
        X = np.ones((S.shape[0], deg + 1))
        for k in range(1, deg + 1):
            X[:, k] = log_S ** k
        return X @ c

    class _Policy:
        __slots__ = ('coeffs', 'n_dates', 'T', 'K', 'deg')

        def __init__(self, coeffs, n_dates, T, K, deg):
            self.coeffs = coeffs
            self.n_dates = n_dates
            self.T = T
            self.K = K
            self.deg = deg

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=float)
            if cs.ndim == 0:
                cs = cs.reshape(1)
            t = time
            if isinstance(t, (list, tuple, np.ndarray)):
                t = float(np.asarray(t, dtype=float).mean())
            else:
                t = float(t)
            
            # Map time to date index (0-based)
            # t is in [0, T]. Dates are at k*T/n_dates for k=1..n_dates.
            # If t is close to 0, it's before first date? 
            # Contract says "no time-0 exercise".
            # If t is very small, we assume it's the first date? Or we shouldn't exercise?
            # Usually, if t < t_1, we can't exercise. But the evaluator likely passes t exactly at exercise dates.
            # Let's round to the nearest date index.
            
            # k = t / (T / n_dates)
            k_float = t / (self.T / self.n_dates)
            k_int = int(round(k_float))
            
            if k_int < 1:
                # Before first exercise date, cannot exercise
                return np.zeros(len(cs), dtype=bool)
            elif k_int > self.n_dates:
                # After maturity, cannot exercise
                return np.zeros(len(cs), dtype=bool)
            
            # k_int is 1..n_dates
            # Index in coeffs is k_int - 1
            idx = k_int - 1
            
            intrinsic = np.maximum(self.K - cs, 0.0)
            
            if idx == self.n_dates - 1:
                # Last date: Exercise if Intrinsic > 0
                # Continuation is 0 (or undefined, effectively 0)
                return intrinsic > 0.0
            else:
                c = self.coeffs[idx]
                if c is None:
                    # Fallback: don't exercise
                    return np.zeros(len(cs), dtype=bool)
                
                # Evaluate continuation value
                log_S = np.log(cs + 1e-10)
                X = np.ones((cs.shape[0], self.deg + 1))
                for k in range(1, self.deg + 1):
                    X[:, k] = log_S ** k
                cont = X @ c
                
                return intrinsic > cont

    return _Policy(coeffs, n_dates, T, K, deg)
