import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        used = 0
        
        archive_F = []

        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(f):
            nonlocal archive_F
            # Remove points dominated by f
            new_F = [af for af in archive_F if not (np.all(f <= af) and np.any(f < af))]
            # Add f if not dominated by any remaining
            if not is_dominated(f, new_F):
                new_F.append(f)
            archive_F = new_F

        # Generate weight vectors for Tchebycheff decomposition
        # 3 objectives -> 2D simplex
        n_weights = 8
        weights = []
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                w = np.array([i, j, k], dtype=float)
                if np.sum(w) > 0:
                    w = w / np.sum(w)
                    weights.append(w)
        np.random.shuffle(weights)
        
        for w in weights:
            if time.time() > deadline - 10 or used >= self.budget:
                break
            
            x0 = lo + np.random.rand(self.dim) * span
            best_f = None
            best_val = float('inf')

            def tcheb_obj(x):
                nonlocal used, best_f, best_val
                if used >= self.budget or time.time() > deadline - 2:
                    return 1e6
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                # Tchebycheff scalarization
                val = float(np.max(w * f))
                if val < best_val:
                    best_val = val
                    best_f = f
                return val

            try:
                minimize(tcheb_obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 40, 'xatol': 1e-4, 'fatol': 1e-6})
                if best_f is not None:
                    update_archive(best_f)
            except Exception:
                pass

        # Fill remaining budget with random sampling
        while used < self.budget and time.time() < deadline - 2:
            x = lo + np.random.rand(self.dim) * span
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            update_archive(f)

        return np.array(archive_F) if archive_F else np.array([lo + span * 0.5])
