"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

class BermudanExercisePolicy:
    """Bermudan put exercise policy using binomial tree backward induction with refined quadrature."""
    
    def __init__(self, K, r, sigma, T, n_exercise_dates, exercise_times):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise_dates = n_exercise_dates
        self.exercise_times = exercise_times
        
        self._compute_thresholds()
    
    def _compute_thresholds(self):
        """Compute critical stock price thresholds for each exercise date via backward induction."""
        n_ex = self.n_exercise_dates
        
        # Fine grid for stock prices
        n_grid = 4001
        S_max = 250.0
        S_min = 0.001
        S_grid = np.linspace(S_min, S_max, n_grid)
        
        # Quadrature for expectation over log-normal transitions
        # Use Gauss-Hermite quadrature for better accuracy
        n_z = 201
        z_max = 5.0
        z_vals = np.linspace(-z_max, z_max, n_z)
        z_weights = norm.pdf(z_vals)
        dz = z_vals[1] - z_vals[0]
        
        # Backward induction
        # V at last exercise date = intrinsic value (no more exercise opportunities)
        V = np.maximum(self.K - S_grid, 0.0)
        
        thresholds = np.zeros(n_ex)
        thresholds[n_ex - 1] = self.K  # At last date, exercise if S < K
        
        for i in range(n_ex - 1, 0, -1):
            t_curr = self.exercise_times[i - 1]
            t_next = self.exercise_times[i]
            dt = t_next - t_curr
            
            if dt <= 0:
                thresholds[i - 1] = thresholds[i]
                continue
            
            # GBM transition: S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            drift = (self.r - 0.5 * self.sigma**2) * dt
            vol_term = self.sigma * np.sqrt(dt)
            
            # Compute S_next for each (S, z) pair
            # S_grid: (n_grid,), z_vals: (n_z,)
            # S_next: (n_grid, n_z)
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            
            # Interpolate V at S_next
            # Clip S_next to grid range for interpolation
            S_next_clipped = np.clip(S_next, S_min, S_max)
            V_next = np.interp(S_next_clipped, S_grid, V, left=0.0, right=0.0)
            
            # Expected continuation value
            cont_val = np.sum(V_next * z_weights[np.newaxis, :] * dz, axis=1)
            
            # Discount
            discount = np.exp(-self.r * dt)
            cont_val = discount * cont_val
            
            # Exercise value
            exercise_val = np.maximum(self.K - S_grid, 0.0)
            
            # Optimal value
            V = np.maximum(exercise_val, cont_val)
            
            # Find threshold: largest S where exercise >= continuation
            diff = exercise_val - cont_val
            positive_mask = diff >= 0
            if np.any(positive_mask):
                last_pos = np.where(positive_mask)[0][-1]
                if last_pos < len(diff) - 1:
                    d0 = diff[last_pos]
                    d1 = diff[last_pos + 1]
                    if d0 - d1 != 0:
                        frac = d0 / (d0 - d1)
                        threshold = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                    else:
                        threshold = S_grid[last_pos]
                else:
                    threshold = S_grid[last_pos]
            else:
                threshold = 0.0
            
            thresholds[i - 1] = threshold
        
        self.thresholds = thresholds
        self.S_grid = S_grid
    
    def exercise(self, current_state, time):
        """Determine whether to exercise at the given time for the given states."""
        idx = np.argmin(np.abs(self.exercise_times - time))
        threshold = self.thresholds[idx]
        return current_state < threshold


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan exercise policy."""
    start_time = time.time()
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    exercise_times = np.linspace(T / n_exercise_dates, T, n_exercise_dates)
    
    policy = BermudanExercisePolicy(K, r, sigma, T, n_exercise_dates, exercise_times)
    
    return policy
# EVOLVE-BLOCK-END
