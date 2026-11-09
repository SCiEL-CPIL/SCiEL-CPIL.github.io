"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # We will use a global evaluation counter
        eval_count = 0

        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            return fx

        # Strategy:
        # 1. Use first ~20 evaluations for a coarse grid/random search to find promising basins.
        # 2. Use remaining evaluations for Nelder-Mead refinement on the best few basins.
        
        # Phase 1: Coarse search
        # f7 is a step-ellipsoidal function. It has a "step" - the function is flat in some regions and steep in others.
        # The minimum is likely in a specific region. 
        # A random search with some structure might help find the basin.
        
        # Let's try a mix of random points and points near 0 (since shift is in [-4,4], 0 is a reasonable guess)
        candidates = []
        
        # 10 random points
        for _ in range(10):
            x = rng.uniform(-5, 5, dim)
            fx = safe_eval(x)
            candidates.append((fx, x))
        
        # 10 points near origin (small radius)
        for _ in range(10):
            x = rng.uniform(-1, 1, dim)
            fx = safe_eval(x)
            candidates.append((fx, x))
            
        # If we have budget left, sort candidates and pick the best few for NM
        candidates.sort(key=lambda t: t[0])
        
        # Select top K candidates for Nelder-Mead
        # We have used 20 evals. Remaining: budget - 20.
        # Let's pick top 3 candidates.
        # Each NM simplex needs dim+1 points. For dim=5, that's 6 points.
        # But we can reuse the candidate as one vertex.
        # So we need dim additional points per simplex.
        # Let's allocate remaining budget among the top 3 candidates.
        
        best_fx = candidates[0][0]
        best_x = candidates[0][1]
        
        if eval_count < budget:
            # Try to refine the best candidate with Nelder-Mead
            # We need to be careful not to exceed budget.
            
            # Let's just do a simple local search around the best point found so far
            # using a shrinking step size, similar to the seed but more aggressive.
            # Actually, the directive says "Multi-start Nelder-Mead".
            # Let's implement a lightweight NM.
            
            # For each of the top 3 candidates:
            # Initialize simplex: center = candidate, vertices = center + delta * e_i
            # delta starts at 0.1
            # Perform NM steps until budget is low or convergence.
            
            top_k = min(3, len(candidates))
            # Distribute remaining budget
            remaining = budget - eval_count
            per_candidate = remaining // top_k if top_k > 0 else 0
            
            for i in range(top_k):
                if eval_count >= budget:
                    break
                
                center = candidates[i][1].copy()
                # Initialize simplex
                # Vertex 0 is center
                # Vertices 1..dim are center + delta * basis[i]
                delta = 0.5
                simplex = [center.copy()]
                for j in range(dim):
                    v = center.copy()
                    v[j] += delta
                    v = np.clip(v, -5, 5)
                    simplex.append(v)
                
                # Evaluate simplex
                fvals = []
                for v in simplex:
                    if eval_count >= budget:
                        break
                    fvals.append(safe_eval(v))
                
                if len(fvals) < dim + 1:
                    continue
                
                # NM iterations
                max_iter = per_candidate // (dim + 1)  # Rough estimate
                for it in range(max_iter):
                    if eval_count >= budget - 1:
                        break
                    
                    # Sort simplex
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_eval(xr)
                    
                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_eval(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        # Accept reflection
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_eval(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                if eval_count >= budget:
                                    break
                                fvals[j] = safe_eval(simplex[j])
                            if eval_count >= budget:
                                break
                    
                    # Update best
                    min_idx = np.argmin(fvals)
                    if fvals[min_idx] < best_fx:
                        best_fx = fvals[min_idx]
                        best_x = simplex[min_idx].copy()
                        
                    # Check convergence
                    if np.std(fvals) < 1e-8:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
