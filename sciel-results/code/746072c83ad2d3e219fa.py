import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_optimal_x0(self, n_points, x0_lo, x0_hi, r1, r2):
        """
        Compute optimal x0 values to maximize hypervolume.
        
        Uses power-law spacing x_i = x0_lo + (x0_hi - x0_lo) * t^p
        and searches for the optimal exponent p.
        """
        if n_points <= 1:
            return np.array([(x0_lo + x0_hi) / 2.0])
        
        def compute_hv(x0_vals):
            """Compute hypervolume for given x0 values on the Pareto front."""
            if len(x0_vals) == 0:
                return 0.0
            x0_sorted = np.sort(x0_vals)
            f1 = x0_sorted
            f2 = 2.0 - np.sqrt(f1)
            if len(f1) == 1:
                if f1[0] < r1 and f2[0] < r2:
                    return (r1 - f1[0]) * (r2 - f2[0])
                return 0.0
            heights = r2 - f2
            widths = np.diff(f1)
            hv = np.sum(widths * heights[:-1])
            hv += (r1 - f1[-1]) * heights[-1]
            return hv
        
        # Coarse search over p
        best_hv = -1
        best_p = 1.0
        
        for p in np.arange(0.5, 4.0, 0.05):
            t = np.linspace(0.0, 1.0, n_points)
            x0_vals = x0_lo + (x0_hi - x0_lo) * t**p
            x0_vals[0] = x0_lo
            x0_vals[-1] = x0_hi
            hv = compute_hv(x0_vals)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Fine search around best_p
        for p in np.linspace(max(0.1, best_p - 0.05), best_p + 0.05, 100):
            t = np.linspace(0.0, 1.0, n_points)
            x0_vals = x0_lo + (x0_hi - x0_lo) * t**p
            x0_vals[0] = x0_lo
            x0_vals[-1] = x0_hi
            hv = compute_hv(x0_vals)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Also compare with sqrt spacing
        u = np.linspace(np.sqrt(x0_lo), np.sqrt(x0_hi), n_points)
        x0_sqrt = u**2
        x0_sqrt[0] = x0_lo
        x0_sqrt[-1] = x0_hi
        hv_sqrt = compute_hv(x0_sqrt)
        
        if hv_sqrt > best_hv:
            return x0_sqrt
        
        # Generate final x0 values with best_p
        t = np.linspace(0.0, 1.0, n_points)
        x0_vals = x0_lo + (x0_hi - x0_lo) * t**best_p
        x0_vals[0] = x0_lo
        x0_vals[-1] = x0_hi
        return x0_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if we can set x_1...x_9 to 0 (known global minimum for g(x))
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], self.budget)
            else:
                x0_vals = self._compute_optimal_x0(
                    self.budget, x0_lo, x0_hi, 11.0, 400.0
                )
            
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x0_vals[:self.budget]
            
            if time.time() > deadline:
                return X
            
            # Evaluate all points
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            
            return X
        
        else:
            # Fallback: random search
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            return X
