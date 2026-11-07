import numpy as np
import time
import scipy.optimize
import scipy.stats.qmc as qmc
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = None
        best_fx = np.inf
        history = []  # (fx, x)

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                x = x[:self.dim]
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            history.append((fx, x.copy()))
            return fx

        dim = self.dim
        bounds = [(-5, 5)] * dim

        # Phase 1: Global Exploration
        # Use Sobol sequence for better coverage
        n_global = 60
        sampler = qmc.Sobol(d=dim, seed=12345)
        points = sampler.random(n_global)
        # Scale from [0,1] to [-5,5]
        points = points * 10.0 - 5.0
        
        # Add center and points near the nominal minimum (1,1,1,1,1)
        points = np.vstack([points, np.zeros(dim), np.ones(dim)])
        
        # Evaluate initial points
        for x0 in points:
            if call_count >= self.budget - 100: # Reserve budget for local
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            tracked_func(best_x)

        # Sort history to get best points
        history.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization from top diverse points
        # Select up to 3 diverse points
        selected_points = []
        for fx, x in history:
            is_diverse = True
            for sel_x in selected_points:
                if np.linalg.norm(x - sel_x) < 1.0:
                    is_diverse = False
                    break
            if is_diverse:
                selected_points.append(x)
                if len(selected_points) >= 3:
                    break
        
        # If we didn't find enough diverse points, just take the best ones
        if len(selected_points) < 3:
            for fx, x in history:
                if x not in selected_points: # Rough check
                    selected_points.append(x)
                if len(selected_points) >= 3:
                    break

        # Run Powell from each selected point
        # Powell is derivative-free and good for Rosenbrock
        maxfev_per_restart = 120
        for x0 in selected_points:
            if time.time() > deadline:
                break
            if call_count >= self.budget - 20:
                break
            
            try:
                res = scipy.optimize.minimize(
                    tracked_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 200,
                        'maxfev': maxfev_per_restart,
                        'xtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
            except TimeoutError:
                break
            except Exception:
                pass

        # Phase 3: Final Polish with L-BFGS-B
        # L-BFGS-B is very fast for the final step if we are close
        if time.time() < deadline and call_count < self.budget - 10:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-15,
                            'eps': 1e-8
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
