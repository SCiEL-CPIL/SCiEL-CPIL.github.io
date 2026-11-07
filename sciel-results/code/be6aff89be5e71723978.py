"""Grid-based adaptive search for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid subdivision with multi-basin tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            f_val = float(func(x))
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Coarse grid 2^dim over [-5,5]^dim
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)
        grid_pts = self._grid(lo, hi, 2)
        
        results = []
        for pt in grid_pts:
            fv = evaluate(pt)
            if np.isinf(fv):
                break
            results.append((fv, pt.copy()))
        
        if not results:
            return best_x, best_f
        
        results.sort(key=lambda x: x[0])
        
        # Phase 2: Refine around top 4 points
        top_k = min(4, len(results))
        for _, pt in results[:top_k]:
            if calls_made >= budget or time.time() > deadline:
                break
            # Define box around pt with width = 2.5 (half of 5)
            box_lo = np.clip(pt - 1.25, -5.0, 5.0)
            box_hi = np.clip(pt + 1.25, -5.0, 5.0)
            # Evaluate 2^dim grid in this box
            sub_grid = self._grid(box_lo, box_hi, 2)
            for pt2 in sub_grid:
                fv = evaluate(pt2)
                if np.isinf(fv):
                    break
        
        # Phase 3: Further refinement around best point
        if best_x is not None and calls_made < budget:
            # Use remaining budget for local search
            remaining = budget - calls_made
            step = 0.5
            for _ in range(min(remaining, 80)):
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for sign in [-1, 1]:
                        if calls_made >= budget or time.time() > deadline:
                            break
                        cand = best_x.copy()
                        cand[d] += sign * step
                        cand = np.clip(cand, -5.0, 5.0)
                        fv = evaluate(cand)
                        if np.isinf(fv):
                            break
                        if fv < best_f:
                            best_f = fv
                            best_x = cand.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                if step < 1e-3:
                    break
        
        return best_x, best_f

    def _grid(self, lo, hi, n_per_dim):
        grid_1d = np.linspace(lo, hi, n_per_dim)
        pts = np.array(np.meshgrid(*[grid_1d] * self.dim)).reshape(-1, self.dim).T
        return pts
# EVOLVE-BLOCK-END
