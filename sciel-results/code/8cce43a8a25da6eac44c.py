"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 45s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Stratified random sampling for good global coverage
        n_strata = 10
        n_initial = min(25, budget // 12)
        candidates = []
        
        # Stratified sampling: divide each dimension into strata
        for i in range(n_initial):
            if eval_count >= budget or not time_ok():
                break
            x = np.zeros(n)
            for d in range(n):
                stratum = int(rng.integers(0, n_strata))
                x[d] = (stratum + rng.random()) / n_strata * 10.0 - 5.0
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))
        
        # Add some pure random points
        n_random = min(5, budget - eval_count)
        for _ in range(n_random):
            if eval_count >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive CMA-ES-like local search on top candidates
        top_k = min(3, len(candidates))
        remaining = budget - eval_count
        per_candidate = max(20, remaining // max(1, top_k))

        for rank in range(top_k):
            if eval_count >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            
            # CMA-ES style adaptive step size search
            sigma = 1.0  # initial step size
            mean = x0.copy()
            best_local = x0.copy()
            best_local_f = f0
            no_improve = 0
            max_iter = per_candidate
            
            for it in range(max_iter):
                if eval_count >= budget or not time_ok():
                    break
                
                # Generate perturbation
                step = sigma * rng.standard_normal(n)
                candidate = np.clip(mean + step, -5, 5)
                fx = evaluate(candidate)
                if fx is None:
                    break
                
                if fx < best_local_f:
                    best_local_f = fx
                    best_local = candidate.copy()
                    mean = candidate.copy()
                    sigma *= 1.05  # slightly increase step on improvement
                    no_improve = 0
                else:
                    no_improve += 1
                    sigma *= 0.95  # decrease step on non-improvement
                    # Pull mean toward best
                    mean = 0.9 * mean + 0.1 * best_local
                
                # Adaptive step size based on recent history
                if no_improve > 10:
                    sigma *= 0.8
                    no_improve = 5
                
                # Convergence check
                if sigma < 1e-6:
                    break
            
            # Update global best
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local.copy()

        # Phase 3: Fine-grained coordinate descent polish
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            coord_sigma = 0.1
            
            for _ in range(min(30, budget - eval_count)):
                if eval_count >= budget or not time_ok():
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * coord_sigma
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy is None:
                            break
                        if fy < f:
                            x, f = y, fy
                            improved = True
                        if eval_count >= budget or not time_ok():
                            break
                    if eval_count >= budget or not time_ok():
                        break
                if not improved:
                    coord_sigma *= 0.5
                    if coord_sigma < 1e-5:
                        break
            
            # Final fine random search
            if eval_count < budget and time_ok():
                fine_sigma = 0.05
                for _ in range(budget - eval_count):
                    if not time_ok() or eval_count >= budget:
                        break
                    y = np.clip(x + fine_sigma * rng.standard_normal(n), -5, 5)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    if fy < f:
                        x, f = y, fy
                        fine_sigma *= 1.05
                    else:
                        fine_sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
