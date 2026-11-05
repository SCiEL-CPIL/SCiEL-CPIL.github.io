import numpy as np
import time as _time
from scipy.optimize import brentq

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
    V = np.zeros((n_exercise, n_paths))
    
    # Regression coefficients for each step (Chebyshev)
    # We store the Chebyshev coefficients and the scaling/offset for mapping
    coeffs = [None] * n_exercise
    # Chebyshev fits are on [-1, 1]. We map S in [0, K] to x in [-1, 1].
    # x = (2*S - K) / K
    # S = (x*K + K) / 2
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        
        # Target for regression: Discounted value of next step
        targets = discount * V[j+1, :]
        
        # Use only in-the-money paths for regression
        itm_mask = S_curr < K
        n_itm = np.sum(itm_mask)
        
        if n_itm < 10:
            S_reg = S_curr
            T_reg = targets
        else:
            S_reg = S_curr[itm_mask]
            T_reg = targets[itm_mask]
        
        # Fit Chebyshev polynomial
        # Map S_reg to [-1, 1]
        x_reg = (2.0 * S_reg - K) / K
        x_reg = np.clip(x_reg, -1.0, 1.0) # Ensure within domain
        
        try:
            # Degree 3 or 4? Let's try 4 for better fit with Chebyshev stability.
            deg = 4
            if len(S_reg) <= deg:
                deg = len(S_reg) - 1
            if deg < 1:
                 deg = 1
            
            # chebfit returns coefficients c0, c1, ... such that p(x) = sum c_i T_i(x)
            c = np.polynomial.chebyshev.chebfit(x_reg, T_reg, deg)
            coeffs[j] = c
        except:
            # Fallback to constant (degree 0)
            coeffs[j] = np.array([np.mean(T_reg)])
            
        # Evaluate continuation value for all paths at current step
        # Map S_curr to [-1, 1]
        x_curr = (2.0 * S_curr - K) / K
        x_curr = np.clip(x_curr, -1.0, 1.0)
        
        cont_values = np.polynomial.chebyshev.chebval(x_curr, coeffs[j])
        # Ensure continuation value is non-negative
        cont_values = np.maximum(cont_values, 0.0)
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
    # Extract Thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last step (j=11), threshold is K
    thresholds[11] = K
    
    for j in range(10, -1, -1):
        if coeffs[j] is None:
            thresholds[j] = K
            continue
            
        c = coeffs[j]
        
        # Define the function to find the root of: f(S) = Payoff(S) - Continuation(S)
        # f(S) = max(K-S, 0) - P(S)
        # We are looking for S in [0, K] where f(S) = 0.
        # For S > K, Payoff is 0, Continuation is likely positive (or 0), so f(S) < 0.
        # For S < 0, Payoff is K-S > K, Continuation is bounded, so f(S) > 0.
        # So there is a root in [0, K].
        
        def f(S):
            # Map S to x for Chebyshev eval
            x = (2.0 * S - K) / K
            x = np.clip(x, -1.0, 1.0)
            cont = np.polynomial.chebyshev.chebval(x, c)
            cont = max(cont, 0.0)
            pay = max(K - S, 0.0)
            return pay - cont
        
        # Grid search to find a bracket [a, b] where sign changes
        S_grid = np.linspace(0, K, 200)
        f_vals = np.array([f(s) for s in S_grid])
        
        # Find index where sign changes from + to -
        # We want the largest S such that we switch from Exercise (f>0) to Hold (f<0)
        # f(S) > 0 means Payoff > Cont, so Exercise.
        # f(S) < 0 means Payoff < Cont, so Hold.
        # So we look for the crossing from + to -.
        
        # If f(0) <= 0, threshold is 0 (always hold? or always exercise? If f(0)<0, Payoff<Cont at S=0, so Hold is better. But Payoff at 0 is K. Cont is usually < K. So f(0) should be > 0. If not, threshold is 0.)
        # If f(K) >= 0, threshold is K (always exercise).
        
        threshold = K # Default
        
        for i in range(len(S_grid) - 1):
            if f_vals[i] > 0 and f_vals[i+1] < 0:
                # Root is between S_grid[i] and S_grid[i+1]
                a, b = S_grid[i], S_grid[i+1]
                try:
                    root = brentq(f, a, b, xtol=1e-8)
                    threshold = root
                except:
                    threshold = (a+b)/2
                break
            elif f_vals[i] < 0 and f_vals[i+1] > 0:
                # This would be a crossing from Hold to Exercise, which is unusual for a put.
                # We want the boundary where we stop exercising.
                # Since we iterate from 0 to K, the first + to - is the exercise boundary.
                # If we see - to +, it implies the function went negative then positive.
                # We should take the LAST + to - crossing?
                # Actually, for a put, Payoff is linear decreasing, Cont is smooth.
                # They should cross once.
                pass
        
        # If no sign change found in grid, check endpoints
        if f_vals[0] < 0:
            threshold = 0.0
        elif f_vals[-1] > 0:
            threshold = K
            
        thresholds[j] = np.clip(threshold, 0, K)
        
    return ExercisePolicy(thresholds)
