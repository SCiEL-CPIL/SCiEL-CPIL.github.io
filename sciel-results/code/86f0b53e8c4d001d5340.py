import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Halton sequence for better initial coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        global_calls = 15
        halton_points = halton_sequence(global_calls, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: CMA-ES-lite with per-dimension adaptive step sizes
        x = best_x.copy()
        fx = best_f
        sigmas = np.ones(dim) * 1.0  # Per-dimension step sizes
        success_count = np.zeros(dim)
        window_size = 10  # Window for success rate calculation
        history = [[] for _ in range(dim)]  # Per-dimension success history

        nm_interval = 35  # Do NM polish every N ES iterations
        es_count = 0
        nm_maxfev = 70

        while calls_made < total_budget and time.time() < deadline:
            # Periodic Nelder-Mead polish
            if es_count > 0 and es_count % nm_interval == 0 and (total_budget - calls_made) > nm_maxfev + 5:
                try:
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={
                                       'maxfev': nm_maxfev,
                                       'xatol': 1e-8,
                                       'fatol': 1e-8,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        x = best_x.copy()
                        fx = best_f
                        # Reset sigmas after NM
                        sigmas = np.ones(dim) * 0.5
                        for d in range(dim):
                            history[d] = []
                            success_count[d] = 0
                except TimeoutError:
                    break
                except Exception:
                    pass

            if time.time() > deadline or calls_made >= total_budget:
                break

            # CMA-ES-lite: generate perturbation with per-dimension sigma
            y = x.copy()
            success = np.zeros(dim, dtype=bool)
            for d in range(dim):
                perturbation = sigma[d] if False else sigmas[d] * rng.standard_normal()
                y[d] += perturbation
                y[d] = np.clip(y[d], -5, 5)

            try:
                fy = wrapper(y)
            except TimeoutError:
                break

            # Update per-dimension success and adapt sigmas
            if fy < fx:
                # Determine which dimensions contributed to success
                for d in range(dim):
                    if y[d] != x[d]:
                        # Simple heuristic: if the move in this direction improved, mark as success
                        # For 1+1 style, we accept/reject the whole step
                        pass
                # In 1+1 ES, the whole step is accepted or rejected
                # But for per-dimension adaptation, we need a different approach
                # Let's use a coordinate-wise approach instead
                
                # Actually, let's use a simpler approach: if the step succeeds,
                # increase all sigmas slightly; if it fails, decrease them
                sigmas *= 1.03
                sigmas = np.clip(sigmas, 0.005, 5.0)
                
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigmas *= 0.95
                sigmas = np.clip(sigmas, 0.005, 5.0)

            es_count += 1

            # Periodic FD gradient polish for rapid local convergence
            if es_count > 0 and es_count % 20 == 0 and (total_budget - calls_made) > 30:
                # Estimate gradient using finite differences
                h = 1e-4
                grad = np.zeros(dim)
                for d in range(dim):
                    xp = x.copy()
                    xp[d] += h
                    xp = np.clip(xp, -5, 5)
                    xm = x.copy()
                    xm[d] -= h
                    xm = np.clip(xm, -5, 5)
                    try:
                        fp = wrapper(xp)
                        fm = wrapper(xm)
                        grad[d] = (fp - fm) / (2 * h)
                    except TimeoutError:
                        break
                    except Exception:
                        grad[d] = 0.0
                
                if time.time() > deadline or calls_made >= total_budget:
                    break

                # If gradient is small enough, do a few steps of gradient descent
                if np.linalg.norm(grad) < 1.0:
                    lr = 0.1
                    for _ in range(5):
                        if time.time() > deadline or calls_made >= total_budget:
                            break
                        x_new = x - lr * grad
                        x_new = np.clip(x_new, -5, 5)
                        try:
                            f_new = wrapper(x_new)
                        except TimeoutError:
                            break
                        except Exception:
                            break
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            # Recompute gradient
                            grad = np.zeros(dim)
                            for d in range(dim):
                                xp = x.copy()
                                xp[d] += h
                                xp = np.clip(xp, -5, 5)
                                xm = x.copy()
                                xm[d] -= h
                                xm = np.clip(xm, -5, 5)
                                try:
                                    fp = wrapper(xp)
                                    fm = wrapper(xm)
                                    grad[d] = (fp - fm) / (2 * h)
                                except TimeoutError:
                                    break
                                except Exception:
                                    grad[d] = 0.0
                            if np.linalg.norm(grad) < 1e-3:
                                break
                        else:
                            lr *= 0.5
                            if lr < 1e-6:
                                break

        # Phase 3: Final high-precision polish
        if calls_made < total_budget - 10 and time.time() < deadline - 5:
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={
                                   'maxfev': min(50, total_budget - calls_made - 5),
                                   'xatol': 1e-10,
                                   'fatol': 1e-10,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
