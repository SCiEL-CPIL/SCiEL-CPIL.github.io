import numpy as np
import time
from scipy.optimize import dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Wrap the function to count calls and track best
        def wrapped_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Use dual_annealing from scipy
        # Bounds: [-5, 5] for each dimension
        bounds = [(-5.0, 5.0)] * dim

        # Parameters for dual_annealing
        # maxiter: maximum number of iterations
        # maxfun: maximum number of function evaluations
        # initial_temp: initial temperature
        # restart_temp_ratio: ratio of temperature after restart
        # visit: number of times to visit each temperature
        # accept: acceptance probability
        # stepsize: step size for the Nelder-Mead part
        # no_of_local: number of local optimization iterations

        # We have 500 calls total.
        # Let's reserve some for safety and initial evaluations.
        # dual_annealing uses a lot of calls.
        # Let's set maxfun to budget - 10 to be safe.
        maxfun = budget - 10

        try:
            result = dual_annealing(
                wrapped_func,
                bounds,
                maxfun=maxfun,
                initial_temp=5200.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                accept=0.658,
                stepsize=0.35,
                no_of_local=1,
                maxiter=100,
                seed=42
            )
        except Exception:
            pass

        # If we have remaining budget, do a local polish with Nelder-Mead or just more random search
        # But dual_annealing should have used most of the budget.
        # Let's check if we have calls left.
        # Actually, dual_annealing stops when maxfun is reached or convergence.
        
        # If we have significant calls left, we can try a simple local search around the best point
        if calls < budget - 10:
            # Simple local search
            x_current = best_x.copy()
            f_current = best_f
            step = 0.5
            for _ in range(20):
                if calls >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                for d in range(dim):
                    for sign in [1, -1]:
                        if calls >= budget - 5:
                            break
                        x_trial = x_current.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = wrapped_func(x_trial)
                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
            if f_current < best_f:
                best_f = f_current
                best_x = x_current

        return best_x, best_f
