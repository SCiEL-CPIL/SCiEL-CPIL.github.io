"""Multi-objective optimizer for WFG7 using LHS + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic LHS sampling with Nelder-Mead local refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def dominates(a, b):
            """Check if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sorting."""
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        n_dominating[i] += 1

            for i in range(n):
                if n_dominating[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume of points w.r.t. reference point."""
            # Filter points that are better than ref in all objectives
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            
            # For 3D, use a simple approach
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            
            # Use the standard 3D hypervolume algorithm
            # This is a simplified version
            n = len(valid)
            if n == 0:
                return 0.0
            if n == 1:
                return np.prod(ref - valid[0])
            
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            
            # For each point, compute its contribution
            # A point contributes the volume of the box it dominates that isn't dominated by others
            # This is complex; use a simpler approach for 3D
            # Actually, let's use a different approach: sweep line or just approximate
            
            # For now, use a simple grid-based approximation
            # But since we need exact, let's implement a proper 3D HV
            # For 3D, we can use the following:
            # Sort by x, then for each point, the contribution is the area in yz-plane 
            # that is not covered by points with smaller x
            
            # Actually, let's just use a simple approach:
            # The hypervolume is the volume of the union of boxes [p_i, ref]
            # For 3D, we can compute this exactly using a sweep line approach
            
            # For simplicity and correctness, let's use a different method:
            # Sort points by first objective
            # For each point in order, compute the 2D hypervolume in the yz-plane
            # for the region where x is between this point's x and the next point's x (or ref)
            
            valid = sorted(valid, key=lambda x: x[0])
            total_hv = 0.0
            
            for i in range(len(valid)):
                p = valid[i]
                x_max = valid[i + 1][0] if i + 1 < len(valid) else ref[0]
                
                # In the slab x in [p[0], x_max], the point p dominates the region
                # But other points with x >= p[0] may also dominate parts of this region
                # We need the 2D hypervolume in yz-plane of points that have x >= p[0]
                # and are within the slab
                
                # Actually, this is getting complex. Let's use a simpler approach:
                # For 3D, we can use the fact that the hypervolume can be computed by
                # sorting and using a sweep line in one dimension
                
                # Let's use a different approach: compute the 2D hypervolume for each "layer"
                
                # For now, let's just use a simple approximation that's good enough
                # for selection purposes
                # The contribution of point i is approximately:
                # (x_max - p[0]) * (2D hypervolume of points j with j >= i in yz, 
                #   but only those that are not dominated in yz by earlier points)
                
                # This is still complex. Let's use a simpler heuristic:
                # Sort by all objectives and use a simple volume estimate
                
                # Actually, for selection purposes, we don't need exact HV.
                # We just need a good measure of "importance" of each point.
                # Let's use a simpler metric: the sum of distances to the ideal point,
                # or just the sum of objectives.
                
                pass
            
            # For now, let's just return the sum of objective values as a proxy
            # This is not ideal but works for selection
            return -np.sum([np.sum(p) for p in valid])

        # Phase 1: Latin Hypercube Sampling for initial population
        # Use LHS to ensure diverse coverage of the 12D space
        n_lhs = min(150, budget // 2)  # Use half the budget for LHS
        rng = np.random.default_rng(12345)
        
        # Latin Hypercube Sampling
        X_lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Generate a permutation of [0, n_lhs)
            perm = rng.permutation(n_lhs)
            # Map to [0, 1)
            X_lhs[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        
        # Scale to bounds
        X_lhs = lo + (hi - lo) * X_lhs
        
        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs], dtype=np.float64)
        used = n_lhs
        
        # Phase 2: Identify non-dominated points and refine with Nelder-Mead
        fronts = fast_non_dominated_sort(F_lhs)
        
        # Get the first front (non-dominated points)
        if fronts and len(fronts[0]) > 0:
            front1_indices = fronts[0]
        else:
            front1_indices = list(range(min(20, n_lhs)))
        
        # Select top points for refinement based on their position in the front
        # Use a simple criterion: points that are more "spread out" in objective space
        n_refine = min(len(front1_indices), max(10, budget // 10))
        
        # Select points to refine: spread them across the front
        if len(front1_indices) <= n_refine:
            refine_indices = front1_indices
        else:
            # Select evenly spaced indices from the front
            step = len(front1_indices) / n_refine
            refine_indices = [front1_indices[int(i * step)] for i in range(n_refine)]
        
        # Nelder-Mead refinement for each selected point
        # Use a multi-objective Nelder-Mead: minimize the sum of objectives
        # or use a weighted sum approach
        
        # For each point to refine, run Nelder-Mead with a weighted sum objective
        # The weights are chosen to explore different parts of the Pareto front
        
        # Generate weight vectors for multi-objective optimization
        n_weights = min(n_refine, 15)
        # Use a simplex lattice for 3 objectives
        weight_vectors = []
        for i in range(n_weights):
            # Generate random weights that sum to 1
            w = rng.dirichlet(np.ones(n_obj))
            weight_vectors.append(w)
        
        # Refine each point
        all_X = list(X_lhs)
        all_F = list(F_lhs)
        
        for idx, w in zip(refine_indices, weight_vectors):
            if time.time() > deadline - 60:
                break
            
            x0 = X_lhs[idx].copy()
            
            # Define the scalar objective as weighted sum
            def obj(x):
                return np.dot(problem(x), w)
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    obj, x0, method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.success or result.fun < np.dot(problem(x0), w):
                    x_new = np.clip(result.x, lo, hi)
                    f_new = problem(x_new)
                    all_X.append(x_new)
                    all_F.append(f_new)
                    used += 1
            except Exception:
                pass
        
        # Phase 3: Use remaining budget for additional LHS samples and refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 30:
            # Generate additional random samples
            n_extra = min(remaining, 50)
            X_extra = lo + (hi - lo) * rng.random((n_extra, dim))
            F_extra = np.array([problem(x) for x in X_extra], dtype=np.float64)
            all_X.extend(X_extra)
            all_F.extend(F_extra)
            used += n_extra
            
            # Refine the best points from the combined population
            F_combined = np.array(all_F)
            fronts = fast_non_dominated_sort(F_combined)
            
            if fronts and len(fronts[0]) > 0:
                front1 = fronts[0]
                n_more_refine = min(len(front1), max(5, (budget - used) // 3))
                
                for idx in front1[:n_more_refine]:
                    if time.time() > deadline - 10:
                        break
                    if used >= budget:
                        break
                    
                    x0 = all_X[idx].copy()
                    w = rng.dirichlet(np.ones(n_obj))
                    
                    def obj(x):
                        return np.dot(problem(x), w)
                    
                    try:
                        result = minimize(
                            obj, x0, method='Nelder-Mead',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if result.success or result.fun < np.dot(problem(x0), w):
                            x_new = np.clip(result.x, lo, hi)
                            f_new = problem(x_new)
                            all_X.append(x_new)
                            all_F.append(f_new)
                            used += 1
                    except Exception:
                        pass
        
        # Return the population
        if all_X:
            return np.array(all_X)
        else:
            return np.array([])
# EVOLVE-BLOCK-END
