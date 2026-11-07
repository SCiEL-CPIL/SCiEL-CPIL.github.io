"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid + multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit with 20s safety margin
        budget = self.budget
        dim = self.dim
        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid search - evaluate all 2^5=32 vertices
        # of the hypercube [-5,5]^5 plus the origin
        vertices = []
        for mask in range(32):
            v = np.array([(5.0 if (mask >> i) & 1 else -5.0) for i in range(dim)])
            vertices.append(v)
        vertices.append(np.zeros(dim))
        
        best_x = None
        best_f = np.inf
        for v in vertices:
            if time.time() > deadline:
                break
            f = eval_point(v)
            if f < best_f:
                best_f = f
                best_x = v.copy()
        
        # Phase 2: Multi-start Nelder-Mead from best vertex and a few other promising points
        # Use remaining budget for local optimization
        remaining = budget - calls
        if remaining <= 0 or best_x is None:
            return best_x, best_f
        
        # Generate a small set of starting points for Nelder-Mead
        starts = [best_x.copy()]
        # Add a few deterministic points near the best vertex
        rng = np.random.default_rng(42)
        for _ in range(min(4, remaining // 10)):
            if time.time() > deadline:
                break
            # Small perturbation around best
            pert = best_x + 0.5 * rng.standard_normal(dim)
            pert = np.clip(pert, -5, 5)
            starts.append(pert)
        
        for start in starts:
            if time.time() > deadline or calls >= budget:
                break
            if remaining <= 0:
                break
            
            # Nelder-Mead simplex method
            # Initialize simplex
            simplex = [start.copy()]
            f0 = eval_point(start)
            if f0 < best_f:
                best_f = f0
                best_x = start.copy()
            
            # Create simplex vertices
            step = 0.5
            for i in range(dim):
                v = start.copy()
                v[i] += step
                v = np.clip(v, -5, 5)
                simplex.append(v)
            
            # Evaluate remaining simplex vertices
            fs = [f0]
            for i in range(1, dim + 1):
                if time.time() > deadline or calls >= budget:
                    break
                f = eval_point(simplex[i])
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
            
            # Nelder-Mead iterations
            max_iter = min(remaining - calls, 50)
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = eval_point(xr)
                
                if fr < fs[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_point(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = eval_point(xc)
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls >= budget:
                                break
                            fs[i] = eval_point(simplex[i])
                
                # Update best
                for i, f in enumerate(fs):
                    if f < best_f:
                        best_f = f
                        best_x = simplex[i].copy()
            
            remaining = budget - calls
        
        return best_x, best_f
# EVOLVE-BLOCK-END
