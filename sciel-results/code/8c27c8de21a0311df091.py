"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


class Optimizer:
    """Halton global sampling + Nelder-Mead local refinement for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        # Phase 1: Global Sampling using Halton sequence
        # Halton provides better space-filling properties than pure random,
        # helping to identify the basin of the global minimum more reliably.
        n_global = 150
        points = self._halton_sequence(n_global, dim)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        candidates = []

        for p in points:
            if time.time() > deadline:
                break
            p = np.clip(p, -5.0, 5.0)
            f_p = func(p)
            call_count += 1
            candidates.append((f_p, p.copy()))
            if f_p < best_f:
                best_f = f_p
                best_x = p.copy()

        # Sort candidates to get top few points
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # We try to refine the top 2 candidates if budget allows.
        # Nelder-Mead is efficient for smooth, rotated functions like Rastrigin.
        
        # Estimate evaluations per Nelder-Mead iteration: ~1 per iteration (new point)
        # Total budget for local search
        remaining = budget - call_count
        
        # Allocate budget: 60% to best, 40% to second best (if it exists and budget allows)
        # But simpler: just run on the best one with most of the budget, 
        # and if it's not good enough, we might have missed the global basin.
        # Given Halton coverage, the best point is likely in the global basin or a close one.
        
        if remaining > 50 and best_x is not None:
            # Use the best point
            x_start = best_x.copy()
            # Nelder-Mead options
            # maxiter is max number of iterations. Each iteration uses 1 function eval.
            # We have 'remaining' evals.
            max_iter = remaining
            
            try:
                res = scipy.optimize.minimize(
                    func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f

    def _halton_sequence(self, n, dim):
        """Generate n points in dim dimensions using Halton sequence, scaled to [-5, 5]."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # Primes: 2, 3, 5, 7, 11, ...
            for i in range(n):
                f = 1.0
                i_val = i + 1
                while i_val > 0:
                    i_val, r = divmod(i_val, base)
                    f /= base
                    points[i, d] += r * f
        # Scale from [0, 1] to [-5, 5]
        points = points * 10.0 - 5.0
        return points
# EVOLVE-BLOCK-END
