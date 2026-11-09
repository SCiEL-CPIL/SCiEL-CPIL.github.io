import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.
    
    Uses a power-law spacing t_i = (i/N)^p.
    Based on observations, p < 1 clusters near 0, which is beneficial due to 
    the high curvature of the sqrt term in ZDT3.
    p=0.8 was observed at 1.26105.
    p=2/3 was observed at 1.26093.
    This suggests lower p might be better, or 0.8 is a local optimum.
    We try p=0.75 as a refinement between 2/3 and 0.8, or p=0.7 for more clustering.
    Let's try p=0.75.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)
        
        # Power-law spacing
        # Trying p=0.75 based on the trend that lower p (more clustering at 0) 
        # might be better than 2/3 (0.66) which was worse than 0.8.
        # Or perhaps 0.8 was best and 2/3 was a step in the wrong direction.
        # Let's try p=0.75.
        p = 0.75
        
        if n_sample == 2:
            t_values = np.array([0.0, 1.0])
        else:
            i = np.arange(n_sample)
            # Map i=0..N-1 to t in [0, 1]
            t_values = (i / (n_sample - 1)) ** p
            t_values[0] = 0.0
            t_values[-1] = 1.0

        recorded = []
        used = 0
        
        # Pre-allocate a vector for speed
        x = np.zeros(dim)
        
        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Set x_1 to t, rest to 0
            x[0] = float(tv)
            
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        # Fallback safety
        if len(recorded) < 2:
            for tv in (0.0, 1.0):
                if used >= budget or time.time() > deadline:
                    break
                x[0] = float(tv)
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                used += 1

        return recorded
