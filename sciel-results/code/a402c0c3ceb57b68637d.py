import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Structured initial sampling using Halton sequence
        # This provides good coverage of the search space with low discrepancy
        n_initial = 50
        n_initial = min(n_initial, max(10, budget // 4))

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        initial_points = []
        for i in range(1, n_initial + 1):
            point = np.array([halton(i, p) for p in [2, 3, 5, 7, 11]])
            point = point * 10.0 - 5.0  # Map [0,1]^5 to [-5,5]^5
            initial_points.append(point)

        # Evaluate initial points
        for pt in initial_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        # Phase 2: Multi-Start Local Optimization using Nelder-Mead
        # Sort initial evaluations to get top candidates
        # We need to track which points gave which values
        # Let's re-evaluate or track them - actually let's just use the best few
        # and add some random points for diversity

        remaining = budget - eval_count
        if remaining > 50:
            # Select top candidates from initial exploration
            # Since we didn't track all values, let's use best_x and add random points
            candidates = [best_x.copy()]

            # Add some random points for diversity
            rng = np.random.default_rng(42)
            n_random = min(10, remaining // 10)
            for _ in range(n_random):
                if eval_count >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-5, 5, size=dim)
                val = counting_func(x_rand)
                if val < best_f:
                    candidates.append(x_rand.copy())

            # Run Nelder-Mead from each candidate
            for x_start in candidates:
                if eval_count >= budget or time.time() > deadline:
                    break

                local_budget = budget - eval_count
                if local_budget < 20:
                    break

                try:
                    result = scipy.optimize.minimize(
                        counting_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget,
                            'maxfev': local_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )

                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()

                except Exception:
                    if time.time() > deadline or eval_count >= budget:
                        break
                    continue

        # Phase 3: CMA-ES for fine-tuning from the best point
        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            try:
                popsize = 15
                max_iter = max(5, remaining // popsize - 5)

                res = scipy.optimize.cma(
                    counting_func,
                    best_x,
                    sigma0=0.2,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-10,
                        'verbose': 0,
                        'bounds': [(-5.0, 5.0)] * dim
                    }
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

            except Exception:
                pass

        # Phase 4: Final Polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining // 3,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )

                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()

            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
