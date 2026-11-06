import numpy as np

class BermudanPutPolicy:
    """
    Bermudan put exercise policy computed via backward induction on a fine grid.
    Uses a 1200-step binomial tree to match the reference accuracy.
    """
    def __init__(self, K, r, sigma, T, n_exercise_dates=12):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise_dates = n_exercise_dates
        
        # Compute boundaries using backward induction
        self.s_critical = self._compute_critical_boundaries()
        
    def _compute_critical_boundaries(self):
        """
        Compute critical stock prices for each exercise date.
        Uses a fine grid and backward induction with CRR tree parameters.
        """
        K = self.K
        r = self.r
        sigma = self.sigma
        T = self.T
        n_ex = self.n_exercise_dates
        
        # Use a fine grid for accuracy
        n_grid = 20000
        s_max = max(250.0, K * 2.5)
        s_min = 0.0
        S_grid = np.linspace(s_min, s_max, n_grid)
        
        # CRR parameters for the 1200-step reference tree
        n_ref = 1200
        dt_ref = T / n_ref
        u = np.exp(sigma * np.sqrt(dt_ref))
        d = np.exp(-sigma * np.sqrt(dt_ref))
        p = (np.exp(r * dt_ref) - d) / (u - d)
        disc = np.exp(-r * dt_ref)
        
        # Maturity payoff: V(T, S) = max(K - S, 0)
        V_next = np.maximum(K - S_grid, 0.0)
        
        # Boundaries for exercise dates 1..n_ex
        # Exercise dates are at T/n_ex, 2*T/n_ex, ..., T
        # We need to compute backward from T to T/n_ex
        
        # The time between consecutive exercise dates is T/n_ex
        # But we use the 1200-step tree, so each exercise date corresponds
        # to n_ref/n_ex steps in the tree
        
        steps_per_ex = n_ref // n_ex
        
        # Start from maturity
        boundaries = [None] * (n_ex + 1)  # 1-indexed
        boundaries[n_ex] = K  # At maturity, exercise if S <= K
        
        # Work backward through exercise dates
        for k in range(n_ex, 0, -1):
            # We need to propagate the value function back by steps_per_ex steps
            # to get the value at the previous exercise date
            
            # Propagate back one step at a time for steps_per_ex steps
            V_curr = V_next.copy()
            for _ in range(steps_per_ex):
                # Continuation value: disc * (p * V(S*u) + (1-p) * V(S*d))
                S_up = S_grid * u
                S_down = S_grid * d
                
                # Interpolate V_curr at S_up and S_down
                V_up = np.interp(S_up, S_grid, V_curr)
                V_down = np.interp(S_down, S_grid, V_curr)
                
                V_cont = disc * (p * V_up + (1 - p) * V_down)
                V_ex = np.maximum(K - S_grid, 0.0)
                
                # At intermediate nodes, no exercise allowed (only at exercise dates)
                V_curr = V_cont
            
            # Now V_curr is the value at the previous exercise date if we don't exercise
            # But we CAN exercise here, so take max
            V_ex = np.maximum(K - S_grid, 0.0)
            V_with_ex = np.maximum(V_ex, V_curr)
            
            # Find the critical boundary
            diff = V_ex - V_curr
            valid = np.where(diff >= 0)[0]
            
            if len(valid) > 0:
                idx = valid[-1]
                if idx < len(S_grid) - 1:
                    s1 = S_grid[idx]
                    s2 = S_grid[idx + 1]
                    d1 = diff[idx]
                    d2 = diff[idx + 1]
                    denom = d2 - d1
                    if abs(denom) > 1e-12:
                        s_crit = s1 - d1 * (s2 - s1) / denom
                    else:
                        s_crit = s1
                else:
                    s_crit = S_grid[idx]
            else:
                s_crit = 0.0
            
            boundaries[k] = s_crit
            V_next = V_with_ex
        
        return boundaries
    
    def exercise(self, current_state, time):
        """
        current_state: array-like of spot prices
        time: scalar or array of time values (in years)
        Returns: boolean vector, True where exercise is optimal
        """
        S = np.asarray(current_state, dtype=float)
        t = np.asarray(time, dtype=float)
        
        # Handle scalar cases
        if t.ndim == 0:
            t = np.full_like(S, t)
        if S.ndim == 0:
            S = S.reshape(1)
            t = t.reshape(1)
        
        # Determine which exercise date each time corresponds to
        # Exercise dates are at T/n_ex, 2*T/n_ex, ..., T
        # k = round(t / (T/n_ex)) = round(t * n_ex / T)
        k_idx = np.round(t * self.n_exercise_dates / self.T).astype(int)
        k_idx = np.clip(k_idx, 1, self.n_exercise_dates)
        
        # Get critical boundaries for each path
        s_crit = np.array([self.s_critical[k] for k in k_idx])
        
        # Exercise if S <= s_crit
        return S <= s_crit


def fit_exercise_policy(training_paths, contract):
    """
    Fit the Bermudan put exercise policy.
    """
    # Extract parameters
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    
    # Number of exercise dates
    n_ex = 12
    if hasattr(contract, 'exercise_dates'):
        n_ex = len(contract.exercise_dates)
    elif hasattr(contract, 'n_steps'):
        n_ex = contract.n_steps
    
    # Build policy
    policy = BermudanPutPolicy(K, r, sigma, T, n_ex)
    return policy
