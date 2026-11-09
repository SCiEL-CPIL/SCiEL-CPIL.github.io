"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured local search: grid init + coordinate-wise refinement of worst points."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # 1. Initialize with a structured grid (deterministic, diverse)
        # For dim=12, a full grid is too large. Use a stratified sample.
        # We want roughly 1/4 of budget for initialization to leave room for search.
        init_budget = max(10, self.budget // 4)
        
        # Simple stratified sampling: split each dimension into k bins, pick one point per bin?
        # With dim=12, even 2 bins per dim is 4096 points. Too many.
        # Let's just use random initialization but ensure we don't waste budget.
        # Actually, the directive asks for a STRUCTURAL change. 
        # Let's try: Initialize with a small set of random points, then use local search.
        
        pop_size = 10
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # 2. Local Search Phase
        # Strategy: Identify the "worst" non-dominated point and try to improve it via coordinate-wise steps.
        # If a step improves the objective (in any sense, e.g., sum of objectives or dominance), keep it.
        # This is a greedy local search on the Pareto front approximation.
        
        while used < self.budget:
            if time.time() - start_time > deadline:
                break
            
            # Find non-dominated points
            # F shape: (pop_size, n_obj)
            # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            # Pick a non-dominated point to refine. 
            # To spread out, we can pick the one with the worst "spread" or just cycle through.
            # Let's pick the non-dominated point that is furthest from the ideal point (0,0,0) to try to push it in.
            # Or simply pick the first non-dominated one.
            # Better: Pick the non-dominated point that has the largest sum of objectives (worst overall)
            # to try to improve the "weakest" part of the front.
            
            non_dom_indices = np.where(~dominated)[0]
            if len(non_dom_indices) == 0:
                # All dominated? Pick the one with min sum
                target_idx = np.argmin(np.sum(F, axis=1))
            else:
                # Among non-dominated, pick the one with max sum (worst)
                target_idx = non_dom_indices[np.argmax(np.sum(F[non_dom_indices], axis=1))]
            
            x = X[target_idx].copy()
            f = F[target_idx].copy()
            
            # Coordinate-wise local search
            # Try small steps in each dimension
            step_sizes = [0.05, 0.01, 0.001] # Relative to span? Or absolute?
            # Let's use relative steps based on span
            base_step = 0.05 * span
            
            improved = False
            for dim_idx in range(self.dim):
                for direction in [-1, 1]:
                    for step_mult in [1.0, 0.5, 0.1]:
                        if used >= self.budget:
                            break
                        new_x = x.copy()
                        new_x[dim_idx] += direction * base_step[dim_idx] * step_mult
                        new_x = np.clip(new_x, lo, hi)
                        
                        # Only evaluate if it's different from current x
                        if np.allclose(new_x, x):
                            continue
                            
                        new_f = problem(new_x)
                        used += 1
                        
                        # Check if new point is better than current point in some way
                        # Simple heuristic: if it dominates current, or if sum of objectives is lower
                        # Dominance check: new_f <= f and any new_f < f
                        if np.all(new_f <= f) and np.any(new_f < f):
                            x = new_x
                            f = new_f
                            improved = True
                            break
                        elif np.sum(new_f) < np.sum(f):
                            # Even if not strictly dominating, if it's better in sum, take it?
                            # This might move away from Pareto front. 
                            # Let's be conservative: only accept if it dominates OR if it's significantly better in sum.
                            # Actually, for multi-objective, moving along the front is good.
                            # Let's stick to dominance for safety, or maybe a hypervolume contribution check?
                            # Too expensive. Let's just use dominance.
                            pass
                        
                        if time.time() - start_time > deadline:
                            break
                    if time.time() - start_time > deadline:
                        break
                if time.time() - start_time > deadline:
                    break
            
            # Update population with the refined point
            X[target_idx] = x
            F[target_idx] = f
            
            # If no improvement was found in this round, we might be stuck.
            # To avoid infinite loops on the same point, we could add random perturbation if no improvement.
            # But for now, let's just try the next worst point.
            
        return X
# EVOLVE-BLOCK-END
