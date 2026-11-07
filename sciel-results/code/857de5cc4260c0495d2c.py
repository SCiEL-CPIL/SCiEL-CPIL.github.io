"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead local search for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 100 local optimizations. Budget is 200, so 2 evals per optimization.
        # This is a very tight budget for Nelder-Mead, but it ensures we hit 100 distinct regions.
        # We will use a simple custom Nelder-Mead implementation or scipy with tight maxiter.
        # Scipy's Nelder-Mead is robust.
        
        # We need to be careful with the number of evaluations.
        # scipy minimize with method='Nelder-Mead' and options={'maxiter': 2, 'maxfev': 2}
        # might not even start properly. 
        # Let's try a simple coordinate ascent/descent or a very simple Nelder-Mead.
        
        # Actually, with 200 budget and 100 starts, we have 2 evals per start.
        # That's just the initial point and one move.
        # Maybe 50 starts with 4 evals each?
        # Or 20 starts with 10 evals each?
        
        # Let's go with 50 starts, 4 evals each.
        n_starts = 50
        evals_per_start = self.budget // n_starts
        
        rng = np.random.default_rng(0)
        
        # Pre-generate start points
        starts = lo + (hi - lo) * rng.random((n_starts, self.dim))
        
        # We will run a simple local search:
        # 1. Evaluate start
        # 2. Try moving in each dimension +/- step
        # 3. Keep best
        
        # Since we have very few evals, a simple grid or coordinate search is better.
        # With 4 evals: 1 start, 3 moves.
        # Let's do:
        # Eval 1: x0
        # Eval 2: x0 + delta
        # Eval 3: x0 - delta
        # Eval 4: x0 + delta2 (if needed)
        
        # But we need to minimize ALL objectives.
        # A scalarizing function? No, we want to record all points.
        # The task says "Every returned objective vector is recorded".
        # So we just need to generate good points.
        
        # Strategy:
        # For each start:
        #   x = start
        #   f = problem(x)
        #   For a few steps:
        #      try random perturbations
        #      if any objective improves, keep?
        #      This is tricky for MO.
        
        # Better: Use a weighted sum scalarization for the local search direction,
        # but record the actual objective vectors.
        # Or just do a random walk that accepts moves if they don't dominate the current point?
        # No, that's slow.
        
        # Let's stick to the simplest effective method:
        # Multi-start random sampling with a small local refinement.
        # Since 200 evals is small, random sampling of 200 points is the baseline.
        # The baseline score is 1.0.
        # Can we do better than random?
        # Yes, by clustering points in promising areas.
        
        # Revised Strategy:
        # 1. Generate 50 random points. Evaluate them. (50 evals)
        # 2. Identify the non-dominated points among these.
        # 3. For each non-dominated point (or top k), perform a local search.
        #    Local search: Try 3 perturbations. Evaluate. Keep best (by sum of objectives or similar).
        
        # Let's implement a simple local search around the initial random points.
        
        pop_size = 50
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Find non-dominated
        nd_mask = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        
        nd_indices = np.where(nd_mask)[0]
        
        # If we have time/budget, refine these
        # We have 150 evals left.
        # Let's pick up to 50 of them (if fewer, pick all) and do 3 evals each for local search.
        # If we have 10 ND points, we can do 15 evals each.
        
        n_refine = min(50, len(nd_indices))
        if n_refine > 0:
            evals_per_refine = (self.budget - used) // n_refine
            if evals_per_refine > 0:
                for idx in nd_indices[:n_refine]:
                    x = X[idx].copy()
                    f = F[idx].copy()
                    # Simple local search
                    for _ in range(evals_per_refine):
                        if time.time() - start_time > max_time:
                            break
                        # Random perturbation
                        step = 0.1 * (hi - lo)
                        perturbation = rng.standard_normal(self.dim) * step
                        x_new = np.clip(x + perturbation, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        
                        # Accept if it doesn't dominate current and is not dominated by current?
                        # Or just accept if sum is lower?
                        # For MO, let's accept if it's not dominated by current.
                        # Actually, we want to explore the front.
                        # Let's just keep the best single point found in this local search
                        # based on a weighted sum with random weights?
                        # No, the evaluator records ALL points.
                        # So we just want to generate as many diverse non-dominated points as possible.
                        # The local search should try to find a point that is better than the start.
                        
                        # Simple heuristic: Accept if sum(f_new) < sum(f)
                        if np.sum(f_new) < np.sum(f):
                            x = x_new
                            f = f_new
                            X[idx] = x
                            F[idx] = f
                        else:
                            # Try a smaller step or different direction?
                            # With limited budget, just try another random direction.
                            pass

        return X

# EVOLVE-BLOCK-END
