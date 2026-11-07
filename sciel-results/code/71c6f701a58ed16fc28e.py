import numpy as np
import time
import scipy.optimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration with Latin Hypercube + key points
        # Use LHS for better coverage in 5D
        n_lhs = 40
        lhs_points = []
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            for i in range(n_lhs):
                if i == 0:
                    lhs_points.append([0.0] * self.dim)
                lhs_points[i][d] = -4.0 + 8.0 * (perm[i] + 0.5) / n_lhs
        
        # Add strategic points
        candidates = list(lhs_points)
        candidates.append(np.zeros(self.dim))
        candidates.append(np.ones(self.dim))
        candidates.append(np.array([1.0] * self.dim))
        candidates.append(rng.uniform(-4, 4, self.dim))
        
        # Evaluate global candidates
        for x0 in candidates:
            if call_count >= self.budget - 20:
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                tracked_func(best_x)
            except:
                pass

        # Phase 2: Local optimization with Powell + adaptive restarts
        bounds = [(-5, 5)] * self.dim
        remaining_budget = self.budget - call_count - 10
        
        if remaining_budget > 10:
            # Powell's method is derivative-free and robust to rotation
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxfev': min(remaining_budget, 200),
                        'xtol': 1e-12,
                        'ftol': 1e-12
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: Adaptive restarts if budget remains
        # If we haven't converged well, try restarting from perturbed best
        if time.time() < deadline and call_count < self.budget - 5:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                # Check if we're near convergence (small improvement in last phase)
                # Try a few restarts with increasing perturbation
                for restart_iter in range(3):
                    if time.time() > deadline or call_count >= self.budget - 5:
                        break
                    if remaining <= 10:
                        break
                    
                    # Perturb best_x with decreasing scale
                    scale = 0.5 / (restart_iter + 1)
                    perturbation = rng.normal(0, scale, self.dim)
                    x_restart = np.clip(best_x + perturbation, -5, 5)
                    
                    try:
                        res = scipy.optimize.minimize(
                            tracked_func,
                            x_restart,
                            method='Powell',
                            bounds=bounds,
                            options={
                                'maxfev': min(remaining, 50),
                                'xtol': 1e-12,
                                'ftol': 1e-12
                            }
                        )
                        remaining = self.budget - call_count - 5
                    except TimeoutError:
                        break
                    except Exception:
                        break

        # Phase 4: Final polish with Nelder-Mead if budget allows
        if time.time() < deadline and call_count < self.budget - 3:
            remaining = self.budget - call_count - 3
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': remaining,
                            'xatol': 1e-12,
                            'fatol': 1e-12
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
