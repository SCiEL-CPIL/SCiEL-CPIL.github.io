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

    # ---- Monte Carlo Longstaff-Schwartz Regression Approach ----
    
    # Ensure training_paths is a numpy array
    # Assume shape is (N_paths, N_times)
    paths = np.asarray(training_paths, dtype=float)
    
    # Determine the number of time steps in the paths
    N_paths, M_times = paths.shape
    
    # Map time indices
    # We need to exercise at n_dates.
    # If M_times == n_dates, we use indices 0 to n_dates-1.
    # If M_times == n_dates + 1, we use indices 1 to n_dates.
    
    if M_times == n_dates:
        start_idx = 0
        end_idx = n_dates
    elif M_times == n_dates + 1:
        start_idx = 1
        end_idx = n_dates + 1
    else:
        # Fallback: assume the last n_dates columns are the exercise dates
        start_idx = M_times - n_dates
        end_idx = M_times
        
    # Extract the relevant spot prices for exercise dates
    # We work backwards from the last date.
    
    # Discount factor per step
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # V holds the optimal value at each step for each path
    V = [None] * (n_dates + 1) # 0..n_dates, we use 1..n_dates
    
    # active[i] is a boolean mask of paths that are still "alive" (not exercised) at step i
    active = [None] * (n_dates + 1)
    
    # Final step (maturity)
    S_final = paths[:, end_idx - 1]
    V[n_dates] = np.maximum(K - S_final, 0.0)
    active[n_dates] = np.ones(N_paths, dtype=bool) # All are active at maturity (they get paid)
    
    # Backward induction
    thresholds = [0.0] * n_dates # thresholds[i-1] for date i
    
    for i in range(n_dates - 1, 0, -1):
        # Current spots
        S_curr = paths[:, start_idx + i - 1]
        
        # Continuation value: discounted future value
        # Only valid for paths that were active in the future
        mask = active[i + 1]
        
        # Get data for fitting
        S_fit = S_curr[mask]
        Y_fit = disc * V[i + 1][mask]
        
        # Fit a polynomial
        # Use degree 3. 
        deg = 3
        try:
            coeffs = np.polyfit(S_fit, Y_fit, deg)
            poly = np.poly1d(coeffs)
        except Exception:
            mean_y = np.mean(Y_fit) if len(Y_fit) > 0 else 0.0
            def poly(s):
                return np.full_like(s, mean_y)
        
        # Calculate Continuation Value for all paths
        Cont_Val = poly(S_curr)
        
        # Calculate Intrinsic Value
        Intrinsic = np.maximum(K - S_curr, 0.0)
        
        # Determine Optimal Value
        V[i] = np.maximum(Intrinsic, Cont_Val)
        
        # Determine who exercises now
        exercised_now = Intrinsic >= Cont_Val
        
        # Update survived mask for next step (i-1)
        active[i] = mask & (~exercised_now)
        
        # Determine Threshold
        # We want S* such that Intrinsic(S*) = Cont_Val(S*)
        # Find the crossover in the fitting data.
        if len(S_fit) == 0:
            thresholds[i - 1] = 0.0
            continue
            
        sort_idx = np.argsort(S_fit)
        S_sorted = S_fit[sort_idx]
        Y_sorted = Y_fit[sort_idx]
        Intrinsic_sorted = np.maximum(K - S_sorted, 0.0)
        
        diff = Intrinsic_sorted - Y_sorted
        
        pos_indices = np.where(diff > 0)[0]
        
        if len(pos_indices) == 0:
            # Always Hold (diff < 0 everywhere)
            star = 0.0
        elif pos_indices[-1] == len(diff) - 1:
            # Always Exercise (diff > 0 everywhere)
            star = K
        else:
            # Find the crossing point
            i_star = pos_indices[-1]
            s1 = S_sorted[i_star]
            s2 = S_sorted[i_star + 1]
            d1 = diff[i_star]
            d2 = diff[i_star + 1]
            
            if abs(d1 - d2) < 1e-10:
                star = s1
            else:
                frac = d1 / (d1 - d2)
                star = s1 + frac * (s2 - s1)
        
        thresholds[i - 1] = star

    class _Policy:
        __slots__ = ('date_thresholds', 'n_dates', 'T')

        def __init__(self, date_thresholds, n_dates, T):
            self.date_thresholds = date_thresholds
            self.n_dates = n_dates
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=float)
            t = time
            if isinstance(t, (list, tuple, np.ndarray)):
                t = float(np.asarray(t, dtype=float).mean())
            else:
                t = float(t)
            # Primary: time is a fraction of the horizon in (0, T].
            # Fallback: if a value > 1 is passed, treat it as a 1..n_dates index.
            if t > 1.0 + 1e-9:
                i = int(round(t))
            else:
                i = int(round(t / self.T * self.n_dates))
            if i < 1:
                i = 1
            elif i > self.n_dates:
                i = self.n_dates
            star = self.date_thresholds[i - 1]
            return cs <= star

    return _Policy(thresholds, n_dates, T)
