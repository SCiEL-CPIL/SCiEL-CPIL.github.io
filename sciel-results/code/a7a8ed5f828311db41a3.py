"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Two-phase optimizer: Differential Evolution for global search, L-BFGS-B for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError()
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Differential Evolution for global exploration
        # Use a moderate population size and max iterations to stay within budget
        # DE with popsize=15 in 5D uses ~75 evaluations per generation
        # We want to use roughly 60-70% of budget for DE
        de_budget = int(budget * 0.65)

        try:
            # Use differential evolution with bounds [-5, 5]^dim
            # maxiter is set high but we control via budget
            # DE in scipy uses (popsize * dim + 1) evals per generation roughly
            # With popsize=12, dim=5: ~61 evals/gen, so ~10 generations for 600 evals
            # We want ~325 evals for DE
            result_de = differential_evolution(
                eval_func,
                bounds=[(-5, 5)] * dim,
                maxiter=50,
                popsize=12,
                tol=0.01,
                seed=42,
                polish=False,  # We'll do our own polish
                init='sobol',
                updating='deferred'
            )
            if result_de.x is not None:
                if result_de.fun < best_f:
                    best_f = result_de.fun
                    best_x = np.clip(result_de.x, -5, 5)
        except TimeoutError:
            pass
        except Exception:
            pass

        # Phase 2: L-BFGS-B local refinement from the best point found
        if best_x is not None and call_count < budget:
            try:
                result_lbfgs = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_lbfgs.fun < best_f:
                    best_f = result_lbfgs.fun
                    best_x = np.clip(result_lbfgs.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more local refinements from DE's top solutions
        # Or try a second DE run with different seed if budget allows
        if best_x is not None and call_count < budget * 0.85:
            try:
                # Try L-BFGS-B from a few perturbed points around best_x
                rng = np.random.default_rng(123)
                for _ in range(5):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x0 = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                    f0 = eval_func(x0)
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    result = minimize(
                        eval_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
