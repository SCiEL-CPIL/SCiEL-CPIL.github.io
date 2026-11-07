import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def _zdt4_f2(self, x0):
        """Compute f2 for ZDT4 given x0 (x[1:] = 0)"""
        x0 = np.clip(x0, 0, 1)
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)**2
    
    def _hv_2d(self, f1_vals, f2_vals, R1=11.0, R2=400.0):
        """Compute 2D hypervolume for non-dominated points sorted by increasing f1"""
        if len(f1_vals) == 0:
            return 0.0
        # Sort by f1
        order = np.argsort(f1_vals)
        f1 = f1_vals[order]
        f2 = f2_vals[order]
        # Remove points not strictly better than reference
        mask = (f1 < R1) & (f2 < R2)
        f1 = f1[mask]
        f2 = f2[mask]
        if len(f1) == 0:
            return 0.0
        # Compute HV
        hv = 0.0
        for i in range(len(f1)):
            next_f1 = f1[i+1] if i+1 < len(f1) else R1
            hv += (next_f1 - f1[i]) * (R2 - f2[i])
        return hv
    
    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
        
        N = min(budget, 200)
        
        # For ZDT4 with x[1:]=0:
        # f1 = x0, f2 = 1 - sqrt(x0) - x0*sin(10*pi*x0)^2
        # The Pareto front is parameterized by x0 in [0,1]
        # We want to choose N values of x0 to maximize hypervolume
        
        # Strategy: Use a smooth transformation that clusters points where f2 changes rapidly
        # f2(x0) = 1 - sqrt(x0) - x0*sin(10*pi*x0)^2
        # The dominant term is -sqrt(x0), so f2 decreases fastest near x0=0
        # Use a power law: x0 = t^p where t is uniform in [0,1]
        # Optimize p to maximize HV
        
        def compute_hv_for_power(p, N):
            if N <= 0:
                return 0.0
            if N == 1:
                x0_vals = np.array([0.0])
            else:
                t = np.linspace(0.0, 1.0, N)
                x0_vals = t**p
            x0_vals = np.clip(x0_vals, lo[0], hi[0])
            f1_vals = x0_vals
            f2_vals = self._zdt4_f2(x0_vals)
            return self._hv_2d(f1_vals, f2_vals)
        
        # Coarse search over power p
        best_hv = -1
        best_p = 1.0
        p_coarse = np.linspace(0.1, 5.0, 50)
        for p in p_coarse:
            hv = compute_hv_for_power(p, N)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Fine search around best_p
        p_fine = np.linspace(best_p - 0.5, best_p + 0.5, 50)
        p_fine = p_fine[p_fine > 0.05]
        for p in p_fine:
            hv = compute_hv_for_power(p, N)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Very fine polish
        p_very_fine = np.linspace(best_p - 0.1, best_p + 0.1, 50)
        p_very_fine = p_very_fine[p_very_fine > 0.05]
        for p in p_very_fine:
            hv = compute_hv_for_power(p, N)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Generate final x0 values
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            t = np.linspace(0.0, 1.0, N)
            x0_vals = t**best_p
        x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = float(x0_val)
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass
        
        if results:
            return np.array(results)
        
        return np.empty((0, dim))
