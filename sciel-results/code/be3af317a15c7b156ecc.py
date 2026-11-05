import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # V[j, i] stores the value of the option at exercise date j (step j+1) for path i
    # We store the "optimal" value (max(payoff, cont)) to use for the previous step's regression target
    V = np.zeros((n_exercise, n_paths))
    
    # Regression coefficients for each step, to be used for threshold extraction
    # Use degree 3 with ITM-only paths for better boundary capture
    poly_degree = 3
    coeffs = [None] * n_exercise
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    # Value is just the payoff
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        
        # Target for regression: Discounted value of next step
        targets = discount * V[j+1, :]
        
        # Use only in-the-money paths for regression (standard LSMC practice)
        # This focuses the fit on the relevant region near the exercise boundary
        itm_mask = S_curr < K
        n_itm = np.sum(itm_mask)
        
        if n_itm < 10:
            # Not enough ITM paths, fall back to all paths
            S_reg = S_curr
            T_reg = targets
        else:
            S_reg = S_curr[itm_mask]
            T_reg = targets[itm_mask]
        
        # Fit polynomial: T = c0 + c1*S + c2*S^2 + c3*S^3
        # np.polyfit returns highest degree first.
        try:
            # Use degree 3 if enough points, else reduce
            deg = min(poly_degree, len(S_reg) - 1)
            if deg < 0:
                 deg = 0
            c = np.polyfit(S_reg, T_reg, deg)
            coeffs[j] = c
        except:
            # Fallback to mean if fit fails
            coeffs[j] = np.array([np.mean(T_reg)])
            deg = 0
            
        # Evaluate continuation value for all paths at current step
        cont_values = np.polyval(coeffs[j], S_curr)
        # Ensure continuation value is non-negative
        cont_values = np.maximum(cont_values, 0.0)
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
    # Extract Thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last step (j=11), threshold is K (exercise if S < K)
    thresholds[11] = K
    
    for j in range(10, -1, -1):
        if coeffs[j] is None:
            thresholds[j] = K
            continue
            
        c = coeffs[j]
        # Find S such that K - S = P(S) for S in [0, K]
        # f(S) = K - S - P(S)
        # We want the root where f changes from positive to negative
        
        # Create a fine grid
        S_grid = np.linspace(0, K, 500)
        f_vals = K - S_grid - np.polyval(c, S_grid)
        
        # Find sign changes
        signs = np.sign(f_vals)
        signs[signs == 0] = 1 
        
        changes = np.where(np.diff(signs) != 0)[0]
        
        if len(changes) > 0:
            # Take the last sign change (highest S where we transition from exercise to hold)
            idx = changes[-1]
            # Linear interpolation for better precision
            x1, x2 = S_grid[idx], S_grid[idx+1]
            y1, y2 = f_vals[idx], f_vals[idx+1]
            if abs(y2 - y1) < 1e-12:
                thresh = x1
            else:
                thresh = x1 - y1 * (x2 - x1) / (y2 - y1)
                
            # Clip to valid range
            thresholds[j] = np.clip(thresh, 0, K)
        else:
            # No sign change
            if f_vals[0] < 0:
                thresholds[j] = 0.0
            else:
                thresholds[j] = K
                
    return ExercisePolicy(thresholds)
