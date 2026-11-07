import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapped_func(x):
            self.calls += 1
            # Deadline guard: Stop if we are too close to the 330s limit
            if time.time() - start_time > 295:
                raise TimeoutError("Time limit exceeded")
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        x0 = np.zeros(self.dim)
        bounds = [(-5, 5)] * self.dim

        def callback(xk):
            # Check time in callback as well, just in case
            if time.time() - start_time > 295:
                raise TimeoutError("Time limit exceeded")
            return False

        try:
            res = minimize(
                wrapped_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfun': self.budget,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'eps': 1e-5
                },
                callback=callback
            )
            
            if hasattr(res, 'x') and res.x is not None:
                # Verify the result doesn't exceed budget (should be handled by maxfun)
                # And update best if necessary
                if self.calls <= self.budget:
                    # res.fun is the value at res.x
                    # But we might have found a better point during the process?
                    # L-BFGS-B returns the best point found.
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x
                else:
                    # If we somehow exceeded budget, use tracked best
                    pass
            else:
                pass

        except TimeoutError:
            # Time limit hit, use best found so far
            pass
        except ValueError:
            # Budget hit, use best found so far
            pass
        except Exception:
            # Any other error, use best found so far
            pass

        if self.best_x is None:
            self.best_x = x0
            # Evaluate x0 if we haven't
            if self.calls == 0:
                self.best_f = func(x0)
                self.calls += 1
            else:
                # If we have calls but no best_x (unlikely), re-evaluate?
                # Or just assume x0 is best
                self.best_f = func(self.best_x)
                self.calls += 1

        return self.best_x, self.best_f
