import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 315s for safety margin
        self.deadline = self.start_time + 315

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        eval_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Initial diverse sampling
        # Use a mix of uniform, centered, and structured sampling
        n_global = 40
        for i in range(n_global):
            if time.time() > self.deadline or eval_count >= budget:
                break
            if i < 20:
                x = rng.uniform(-5, 5, dim)
            elif i < 30:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: L-BFGS-B from the best point
        remaining_budget = budget - eval_count
        if remaining_budget > 10 and time.time() < self.deadline:
            try:
                res = scipy.optimize.minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfun': min(remaining_budget, 150), 'maxiter': 80}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: CMA-ES for more robust local search
        remaining_budget = budget - eval_count
        if remaining_budget > 30 and time.time() < self.deadline:
            try:
                from scipy.optimize import minimize

                # Use a smaller step size for CMA-ES
                sigma0 = 1.0
                cma_options = {
                    'maxiter': 50,
                    'bounds': bounds,
                    'verbose': 0,
                    'seed': 42
                }

                # We'll use a custom CMA-ES implementation via scipy's minimize with method='Nelder-Mead'
                # or we can try a simple coordinate descent refinement

                # Try Nelder-Mead for a quick local refinement
                res = scipy.optimize.minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 4: Perturbation-based search with local refinement
        remaining_budget = budget - eval_count
        if remaining_budget > 20 and time.time() < self.deadline:
            n_perturb = 8
            for i in range(n_perturb):
                if time.time() > self.deadline or eval_count >= budget - 5:
                    break

                # Decreasing step sizes
                step = 2.0 / (i + 1)
                x_pert = best_x + rng.normal(0, step, dim)
                x_pert = np.clip(x_pert, -5, 5)

                try:
                    counted_func(x_pert)
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    continue

                # If improved, do a quick local search
                if eval_count < budget - 10 and time.time() < self.deadline:
                    try:
                        res = scipy.optimize.minimize(
                            counted_func,
                            x_pert,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxfun': min(budget - eval_count - 3, 40), 'maxiter': 20}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                    except (TimeoutError, RuntimeError):
                        break
                    except Exception:
                        continue

        # Phase 5: Final fine-tuning with very small steps
        remaining_budget = budget - eval_count
        if remaining_budget > 5 and time.time() < self.deadline:
            n_fine = min(remaining_budget, 20)
            for i in range(n_fine):
                if time.time() > self.deadline or eval_count >= budget:
                    break
                step = 0.1 / (i + 1)
                x_fine = best_x + rng.normal(0, step, dim)
                x_fine = np.clip(x_fine, -5, 5)
                try:
                    counted_func(x_fine)
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    continue

        return best_x, best_f
