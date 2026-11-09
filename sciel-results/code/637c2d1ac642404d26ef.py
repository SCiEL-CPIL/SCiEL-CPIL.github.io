"""Multi-start adaptive (1+1)-ES with interleaved Nelder-Mead and final polish for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with interleaved Nelder-Mead and final polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse Initialization using Halton sequence
        n_init = 35
        init_results = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.05, 0.05, dim)
            point = np.clip(point, -5, 5)

            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        # Sort by fitness
        init_results.sort(key=lambda x: x[1])

        # Phase 2: Adaptive (1+1)-ES from top starts with interleaved Nelder-Mead
        K = min(3, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(30, remaining_budget // K)

        for x0, f0 in top_starts:
            if time.time() > deadline or calls_made >= budget:
                break

            # --- Adaptive (1+1)-ES phase ---
            x = x0.copy()
            fx = f0
            sigma = 1.5
            stagnation = 0
            max_stagnation = 20
            success_window = []
            window_size = 8
            es_calls = 0
            max_es_calls = budget_per_start // 2

            while es_calls < max_es_calls:
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                es_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    sigma *= 1.3
                    stagnation = 0
                else:
                    success_window.append(0)
                    sigma *= 0.85
                    stagnation += 1

                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window)
                    if success_rate > 0.4:
                        sigma *= 1.15
                    elif success_rate < 0.1:
                        sigma *= 0.65
                    success_window = success_window[-window_size // 2:]

                sigma = np.clip(sigma, 0.003, 10.0)

                if stagnation >= max_stagnation:
                    if rng.random() < 0.5:
                        x_new = np.clip(best_x + rng.standard_normal(dim) * 1.2, -5, 5)
                    else:
                        x_new = rng.uniform(-4, 4, dim)
                    fx_new = evaluate(x_new)
                    es_calls += 1
                    if fx_new < best_f:
                        best_x = x_new.copy()
                        best_f = fx_new
                    x = x_new.copy()
                    fx = fx_new
                    stagnation = 0
                    sigma = 1.2
                    success_window = []

            # --- Nelder-Mead refinement (interleaved, right after ES) ---
            if time.time() > deadline or calls_made >= budget:
                continue

            x_nm = x.copy()
            f_nm = fx

            delta = 0.3
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [f_nm]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i + 1])
                simplex_f.append(f_new)

            if len(simplex_f) < dim + 1:
                continue

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = min(budget_per_start // 3, 30)
            no_improve_count = 0

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break

                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                max_dist = 0.0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-6:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    no_improve_count = 0
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    no_improve_count = 0
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        no_improve_count = 0
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve_count += 1

                if no_improve_count > 5:
                    break

        # Phase 3: Final polish on global best with tight local search
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.03
            polish_calls = 0
            max_polish = min(50, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                polish_calls += 1

                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 1.05
                else:
                    sigma_p *= 0.75

                sigma_p = np.clip(sigma_p, 0.0005, 0.3)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
