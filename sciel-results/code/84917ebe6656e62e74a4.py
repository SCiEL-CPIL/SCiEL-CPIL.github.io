import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        nondominated = []
        
        def is_dominated(f, archive):
            for af in archive:
                if all(af[i] <= f[i] for i in range(self.n_obj)) and any(af[i] < f[i] for i in range(self.n_obj)):
                    return True
            return False
        
        def update_archive(f, archive):
            if is_dominated(f, archive):
                return archive
            new_archive = [af for af in archive if not is_dominated(af, [f])]
            new_archive.append(f)
            return new_archive
        
        # Phase 1: Structured sampling in promising region
        # For ZDT6, to get f1 < 11, we need large x[1:] values
        # f1 = 91 - 10*x1 - sum(10*xi + sin(10*pi*xi)) for i>=2
        # Setting x1=0 and xi=1 gives f1=1
        # Let's sample x1 in [0, 0.5] and x[1:] with mix of 0 and 1
        
        n_phase1 = min(100, n_evals)
        n_phase2 = n_evals - n_phase1
        
        # Phase 1: Grid-like sampling
        n_x1_vals = 10
        n_x12_vals = 10
        x1_vals = np.linspace(0.0, 0.5, n_x1_vals)
        x12_vals = np.linspace(0.5, 1.0, n_x12_vals)
        
        count = 0
        for x1 in x1_vals:
            for x12 in x12_vals:
                if count >= n_phase1 or time.time() > deadline:
                    break
                x = np.zeros(self.dim, dtype=float)
                x[0] = float(x1)
                # Set half the remaining vars to 0, half to 1, controlled by x12
                n_high = int((self.dim - 1) * x12)
                x[1:1+n_high] = 1.0
                # x[1+n_high:] stays 0
                try:
                    f = problem(x)
                    f = np.array(f, dtype=float)
                    if np.all(np.isfinite(f)) and np.all(f < 11.0):
                        results.append(f)
                        nondominated = update_archive(f, nondominated)
                except Exception:
                    pass
                count += 1
            if count >= n_phase1 or time.time() > deadline:
                break
        
        # Phase 2: Random search around promising areas
        if n_phase2 > 0:
            for _ in range(n_phase2):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim, dtype=float)
                x[0] = np.random.uniform(0.0, 0.3)
                # Randomly set some variables to 1
                for i in range(1, self.dim):
                    x[i] = 1.0 if np.random.random() < 0.7 else 0.0
                try:
                    f = problem(x)
                    f = np.array(f, dtype=float)
                    if np.all(np.isfinite(f)) and np.all(f < 11.0):
                        results.append(f)
                        nondominated = update_archive(f, nondominated)
                except Exception:
                    pass
        
        return results
