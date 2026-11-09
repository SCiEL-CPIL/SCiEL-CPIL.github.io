"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG4 using grid-based decomposition
    with Latin Hypercube Sampling for maximal spatial coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use a fraction of budget for initial exploration
        n_lhs = min(60, budget // 3)
        lhs_points = self._latin_hypercube(n_lhs, dim, lo, hi)
        
        archive = []
        eval_count = 0

        def evaluate_and_add(x):
            nonlocal eval_count
            if eval_count >= budget:
                return
            f = np.array(problem(x), dtype=float)
            archive.append((f, x.copy()))
            eval_count += 1

        # Evaluate LHS points
        for x in lhs_points:
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            evaluate_and_add(x)

        # Phase 2: Refinement around promising regions
        # Identify non-dominated points and refine around them
        if archive:
            # Get non-dominated points
            nd_points = self._get_non_dominated(archive)
            
            # Allocate remaining budget for local refinement
            remaining = budget - eval_count
            if remaining > 0 and nd_points:
                n_refine = min(remaining, len(nd_points) * 3)
                per_point = max(1, n_refine // len(nd_points))
                
                for f, x in nd_points[:n_refine // per_point + 1]:
                    if eval_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Local perturbation around this point
                    for _ in range(per_point):
                        if eval_count >= budget:
                            break
                        if time.time() > deadline:
                            break
                        # Small random perturbation
                        scale = 0.05 * (hi - lo)
                        x_new = x + scale * np.random.randn(dim)
                        x_new = np.clip(x_new, lo, hi)
                        evaluate_and_add(x_new)

        # Phase 3: If budget remains, do additional random sampling in promising regions
        if eval_count < budget:
            # Sample around the best non-dominated points
            if archive:
                nd_points = self._get_non_dominated(archive)
                if nd_points:
                    # Weighted sampling closer to non-dominated points
                    while eval_count < budget:
                        if time.time() > deadline:
                            break
                        # Pick a random non-dominated point and perturb
                        f, x = nd_points[np.random.randint(len(nd_points))]
                        scale = 0.1 * (hi - lo)
                        x_new = x + scale * np.random.randn(dim)
                        x_new = np.clip(x_new, lo, hi)
                        evaluate_and_add(x_new)
            else:
                # Fallback: pure random
                while eval_count < budget:
                    if time.time() > deadline:
                        break
                    x = lo + np.random.rand(dim) * (hi - lo)
                    evaluate_and_add(x)

        return archive

    def _latin_hypercube(self, n, dim, lo, hi):
        """Generate n points using Latin Hypercube Sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Create stratified samples
            indices = np.arange(n)
            np.random.shuffle(indices)
            samples = (indices + np.random.rand(n)) / n
            points[:, d] = lo[d] + samples * (hi[d] - lo[d])
        return points

    def _get_non_dominated(self, archive):
        """Return list of non-dominated (f, x) pairs from archive."""
        if not archive:
            return []
        
        n = len(archive)
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            fi = archive[i][0]
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                fj = archive[j][0]
                # Check if i dominates j
                if np.all(fi <= fj) and np.any(fi < fj):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(fj <= fi) and np.any(fj < fi):
                    dominated[i] = True
                    break
        
        return [archive[i] for i in range(n) if not dominated[i]]
# EVOLVE-BLOCK-END
