import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin before 300s limit

        n = self.dim
        budget = self.budget

        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0

        class _BudgetExceeded(Exception):
            pass

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            f_val = func(x)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with structured sampling
        n_init = min(30, budget)
        safe_func(np.zeros(n))

        for i in range(1, n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            # Mix of random, axis-aligned, and diagonal points
            if i % 3 == 0:
                # Axis-aligned perturbations
                axis = i % n
                x = np.zeros(n)
                x[axis] = rng.uniform(-4, 4)
            elif i % 3 == 1:
                # Diagonal points
                sign = 1 if i % 2 == 0 else -1
                scale = 1.0 + 2.0 * (i % 5) / 4.0
                x = np.full(n, sign * scale)
            else:
                # Random points
                x = rng.uniform(-4.5, 4.5, n)
            safe_func(x)

        if best_f == np.inf:
            return best_x, 0.0

        # Phase 2: Nelder-Mead with adaptive restarts
        def nelder_mead(x0, max_iter, initial_step=0.5):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline or eval_count >= budget:
                return

            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                v = x0.copy()
                v[i] += initial_step
                simplex.append(v)

            # Evaluate simplex
            fvals = []
            for v in simplex:
                if time.time() > deadline or eval_count >= budget:
                    return
                fv = safe_func(v)
                fvals.append(fv)
            fvals = np.array(fvals)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iter):
                if time.time() > deadline or eval_count >= budget:
                    return

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Check convergence
                if np.std(fvals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)

                if time.time() > deadline or eval_count >= budget:
                    return
                f_ref = safe_func(x_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if time.time() > deadline or eval_count >= budget:
                        return
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if time.time() > deadline or eval_count >= budget:
                        return
                    f_con = safe_func(x_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if time.time() > deadline or eval_count >= budget:
                                return
                            fvals[i] = safe_func(simplex[i])

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

        # Run Nelder-Mead from best point found
        nm_iters = min(150, (budget - eval_count) // (n + 1))
        if nm_iters > 5:
            nelder_mead(best_x.copy(), nm_iters, initial_step=0.3)

        # Phase 3: Local refinement with smaller steps
        remaining = budget - eval_count
        if remaining > 10 and time.time() < deadline - 5:
            # Use remaining budget for fine-tuning
            polish_sigma = 0.1
            max_polish = min(remaining // 2, 80)
            for _ in range(max_polish):
                if time.time() > deadline - 5 or eval_count >= budget:
                    break
                x = np.clip(best_x + rng.normal(0, polish_sigma, n), -5, 5)
                safe_func(x)
                polish_sigma *= 0.95

        return best_x, best_f
