"""HYBRID: Rastrigin-aware grid basin identification with DE refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    """Hybrid optimizer: coarse grid to find Rastrigin basins, then DE refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit (evaluator kills at 330s)

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        dim = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid sampling to identify promising basins
        # Büche-Rastrigin has periodic structure; use a grid with spacing ~2.0
        # to sample multiple periods across [-5, 5]
        # Grid points: -5, -3, -1, 1, 3, 5 -> 6 points per dim, but that's 6^5=7776 (too many)
        # Use 4 points per dim: -5, -1.67, 1.67, 5 -> 4^5 = 1024 (still a lot)
        # Use 3 points per dim: -5, 0, 5 -> 3^5 = 241 (manageable, ~48% of budget)
        
        # Actually, let's be smarter. Use a Latin Hypercube-like approach for the first phase
        # to get better coverage with fewer points, then refine the best few.
        
        # Phase 1: Sample ~150 points using Latin Hypercube for good coverage
        n_phase1 = 150
        if max_calls < n_phase1 + 50:
            n_phase1 = max(10, max_calls - 50)
        
        # Latin Hypercube sampling
        indices = [rng.permutation(n_phase1) for _ in range(dim)]
        lhc_points = np.array([
            (indices[d] + rng.random()) / n_phase1 * 10.0 - 5.0
            for d in range(dim)
        ]).T
        
        # Evaluate all LHC points
        for i in range(n_phase1):
            if time.time() > deadline or call_count >= max_calls:
                break
            x = lhc_points[i]
            evaluate(x)
        
        if time.time() > deadline or call_count >= max_calls:
            return best_x, best_f

        # Phase 2: Identify top K basins from Phase 1
        # We need to re-evaluate or track which points were good
        # Since we don't store all points, let's do a second pass with fewer, targeted points
        # around the best regions
        
        # Actually, let's use a different strategy: 
        # Use the best point from Phase 1 as the center for a local DE search
        # and also try a few random restarts
        
        # Phase 2: Local DE refinement around the best point found
        # Use a smaller population and fewer iterations
        remaining_budget = max_calls - call_count
        if remaining_budget < 30:
            # Not enough budget for DE, just do a simple coordinate descent
            x_current = best_x.copy()
            f_current = best_f
            step = 0.5
            for _ in range(remaining_budget // 2):
                if time.time() > deadline or call_count >= max_calls:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline or call_count >= max_calls:
                        break
                    for direction in [1.0, -1.0]:
                        x_trial = x_current.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = evaluate(x_trial)
                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
            return best_x, best_f

        # Phase 2: DE around best point with tight bounds
        # Use bounds of best_x ± 2.0 to stay in the basin
        bounds_local = []
        for d in range(dim):
            lower = max(-5.0, best_x[d] - 2.0)
            upper = min(5.0, best_x[d] + 2.0)
            bounds_local.append((lower, upper))
        
        # Configure DE for local refinement
        # Use remaining budget efficiently
        # popsize=10, maxiter such that popsize * maxiter <= remaining_budget
        popsize = 10
        maxiter = max(1, remaining_budget // popsize - 5)
        
        try:
            differential_evolution(
                evaluate,
                bounds_local,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-10,
                seed=42,
                init='latinhypercube'
            )
        except Exception:
            pass

        if time.time() > deadline or call_count >= max_calls:
            return best_x, best_f

        # Phase 3: If we have budget left, try a few more random restarts with DE
        remaining_budget = max_calls - call_count
        if remaining_budget > 50:
            # Try 1-2 more random restarts
            n_restarts = min(2, remaining_budget // 60)
            for r in range(n_restarts):
                if time.time() > deadline or call_count >= max_calls:
                    break
                
                # Random restart point
                x_start = rng.uniform(-5, 5, dim)
                f_start = evaluate(x_start)
                
                if f_start >= best_f:
                    # This restart is worse, skip DE for it
                    continue
                
                # DE around this new point
                bounds_restart = []
                for d in range(dim):
                    lower = max(-5.0, x_start[d] - 1.5)
                    upper = min(5.0, x_start[d] + 1.5)
                    bounds_restart.append((lower, upper))
                
                remaining_budget = max_calls - call_count
                if remaining_budget < 30:
                    break
                
                popsize_r = 8
                maxiter_r = max(1, remaining_budget // popsize_r - 5)
                
                try:
                    differential_evolution(
                        evaluate,
                        bounds_restart,
                        maxiter=maxiter_r,
                        popsize=popsize_r,
                        tol=1e-10,
                        seed=42 + r,
                        init='latinhypercube'
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
