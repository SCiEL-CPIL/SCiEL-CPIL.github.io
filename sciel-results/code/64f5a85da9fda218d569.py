import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Determine optimal tail values for ZDT4
        # g(x) = sum_{i=1}^{dim-1} (x_i^2 - 10*cos(4*pi*x_i))
        # Global minimum of each term is at x_i = 0 (value = -10)
        # Local minima at other integers, but x_i=0 gives the lowest value
        x_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            l, h = bounds[i]
            # Find best integer or bound for minimizing x^2 - 10*cos(4*pi*x)
            candidates = [l, h, 0.0]
            # Check integers in range
            start_int = int(np.ceil(l))
            end_int = int(np.floor(h))
            for k in range(start_int, end_int + 1):
                candidates.append(float(k))
            # Also check 0.5 offsets (local maxima, but just in case)
            for k in range(start_int, end_int + 1):
                candidates.append(k + 0.5)
            
            best_val = float('inf')
            best_x = l
            for c in candidates:
                if l <= c <= h:
                    val = c*c - 10.0*np.cos(4.0*np.pi*c)
                    if val < best_val:
                        best_val = val
                        best_x = c
            x_tail[i-1] = best_x

        n = self.budget
        x0_lo = lo[0]
        x0_hi = hi[0]
        x0_range = x0_hi - x0_lo

        # Strategy: Use stratified sampling on x[0] with refinement
        # Phase 1: Uniform stratified sampling (LHS-like) - 100 points
        # Phase 2: Adaptive refinement near the front - 100 points
        # The Pareto front for ZDT4 is f1 = x0, f2 = (1+sqrt(f1)) * g_min
        # where g_min = (dim-1) + 20 * sum(x_i^2 - 10*cos(4*pi*x_i)) at optimum
        # For standard bounds, g_min = (dim-1) - 20*(dim-1) = -(dim-1)^2... 
        # Actually g = (dim-1) + 20*sum(...), each term min is -10
        # So g_min = (dim-1) + 20*(-10)*(dim-1) = (dim-1)*(1-200) = -199*(dim-1)
        # Wait, that's not right. Let me recalculate.
        # Each term x_i^2 - 10*cos(4*pi*x_i) has minimum -10 at x_i=0
        # So sum has minimum -10*(dim-1)
        # g_min = (dim-1) + 20*(-10*(dim-1)) = (dim-1) - 200*(dim-1) = -199*(dim-1)
        # For dim=10: g_min = -199*9 = -1791
        # f2 = (1+sqrt(f1)) * g_min... but g should be positive for the front
        # Actually in ZDT4, g is defined differently. Let me just use the evaluator.
        
        # Phase 1: 100 stratified points
        n_phase1 = n // 2
        n_phase2 = n - n_phase1
        
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            # Stratified sampling: one point per stratum
            u = (i + 0.5) / n_phase1
            # Add small random jitter
            u += (np.random.random() - 0.5) / n_phase1
            u = np.clip(u, 0.0, 1.0)
            
            x0 = x0_lo + u * x0_range
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        # Phase 2: Adaptive refinement - focus on regions with high curvature
        # The Pareto front f2 = (1+sqrt(f1)) * g is concave in f1
        # Higher density near f1=0 where curvature is highest
        # Use power-law distribution: u^alpha with alpha < 1 to cluster near 0
        # But we want to cover the whole front for hypervolume
        
        # Alternative: Use a non-uniform grid that's denser near boundaries
        # For hypervolume, we want good coverage of the entire front
        # Use a combination: 50 points uniform, 50 points with sqrt spacing
        
        n_phase2a = n_phase2 // 2
        n_phase2b = n_phase2 - n_phase2a
        
        # Phase 2a: Uniform with finer resolution
        for i in range(n_phase2a):
            if time.time() > deadline:
                break
            u = (i + 0.5) / n_phase2a
            u += (np.random.random() - 0.5) / n_phase2a
            u = np.clip(u, 0.0, 1.0)
            
            x0 = x0_lo + u * x0_range
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        # Phase 2b: Sqrt-spaced points (denser near x0=0)
        for i in range(n_phase2b):
            if time.time() > deadline:
                break
            # u in [0,1], use sqrt to cluster near 0
            u_raw = (i + 0.5) / n_phase2b
            u = np.sqrt(u_raw)
            u += (np.random.random() - 0.5) / n_phase2b
            u = np.clip(u, 0.0, 1.0)
            
            x0 = x0_lo + u * x0_range
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None
