import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    dt = T / n_exercises
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # Grid setup
    # Using a log-uniform grid for better resolution at low S
    S_min = 0.01
    S_max = 200.0
    n_grid = 100000  # Increased resolution
    
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # V at maturity (step 12)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # boundaries[i] for i in 1..12
    # boundaries[12] is the maturity boundary (exercise if S <= K)
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K
    
    # Backward induction from step 11 down to 1
    for j in range(n_exercises - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break: fill remaining boundaries with maturity boundary
            for jj in range(j, 0, -1):
                boundaries[jj] = boundaries[n_exercises]
            break
            
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next using PCHIP for higher accuracy
        # PCHIP preserves monotonicity, which is desirable for option values
        pchip_func = PchipInterpolator(S_grid, V_next)
        
        # Evaluate at Su and Sd
        # Clip Su and Sd to grid range to avoid extrapolation issues, 
        # though geomspace covers the likely range well.
        # np.clip to [S_min, S_max]
        Su_clipped = np.clip(Su, S_min, S_max)
        Sd_clipped = np.clip(Sd, S_min, S_max)
        
        V_up = pchip_func(Su_clipped)
        V_down = pchip_func(Sd_clipped)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        # Find boundary
        diff = intrinsic - continuation
        
        # diff is positive for small S, negative for large S.
        # Find the index where sign changes.
        exercise_indices = np.where(diff > 0)[0]
        
        if len(exercise_indices) > 0:
            last_idx = exercise_indices[-1]
            if last_idx < n_grid - 1:
                # Linear interpolation for the boundary
                d0 = diff[last_idx]
                d1 = diff[last_idx + 1]
                if d1 != d0:
                    t_interp = d0 / (d0 - d1)
                    boundaries[j] = S_grid[last_idx] + t_interp * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[j] = S_grid[last_idx]
            else:
                boundaries[j] = S_grid[-1]
        else:
            boundaries[j] = S_min
        
        V_next = V_current
    
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
            if isinstance(time, (int, float)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                times = np.asarray(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                # Vectorized boundary lookup
                b = np.array([self.boundaries[t] for t in times_clipped])
                return current_state <= b
    
    return ExercisePolicy(boundaries, K)
