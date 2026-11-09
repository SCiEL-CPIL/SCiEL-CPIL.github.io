"""
Optimizer for Bent Cigar (f12) combining Sobol global search with
Adaptive (1+1)-ES local refinement for anisotropic landscapes.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf

        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")

            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)

            calls_used += 1

            # Evaluate
            fx = func(x)

            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        n_global = 32
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)

        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0

        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break

        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return None, np.inf
        else:
            x0 = best_x.copy()

        # --- Phase 2: Local Refinement with (1+1)-ES ---
        # (1+1)-ES with sigma adaptation is well-suited for anisotropic
        # functions like Bent Cigar: it adapts step size based on success/failure
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            x_cur = x0.copy()
            f_cur = best_fx
            sigma = 0.5  # Initial step size
            success = 0
            attempts = 0

            while calls_used < budget and time.time() < deadline:
                # Generate candidate
                z = np.random.randn(dim)
                x_trial = x_cur + sigma * z
                # Clip to box
                x_trial = np.clip(x_trial, -5.0, 5.0)

                try:
                    f_trial = wrapped_obj(x_trial)
                except Exception:
                    break

                attempts += 1

                if f_trial < f_cur:
                    # Success: move to trial, increase sigma
                    x_cur = x_trial
                    f_cur = f_trial
                    sigma *= 1.5
                    success += 1
                else:
                    # Failure: decrease sigma
                    sigma *= 0.87

                # If sigma becomes too small, break
                if sigma < 1e-12:
                    break

                # Periodically update global best
                if f_cur < best_fx:
                    best_fx = f_cur
                    best_x = x_cur.copy()

            # --- Phase 3: High-precision polish pass ---
            # If we have budget remaining, do a final tight local search
            remaining = budget - calls_used
            if remaining > 5 and time.time() < deadline:
                sigma_final = max(sigma * 0.1, 1e-8)
                x_polish = best_x.copy()
                f_polish = best_fx
                for _ in range(min(remaining, 50)):
                    if time.time() >= deadline or calls_used >= budget:
                        break
                    z = np.random.randn(dim)
                    x_trial = x_polish + sigma_final * z
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    try:
                        f_trial = wrapped_obj(x_trial)
                    except Exception:
                        break
                    if f_trial < f_polish:
                        x_polish = x_trial
                        f_polish = f_trial
                        sigma_final *= 1.1
                    else:
                        sigma_final *= 0.8
                    if f_polish < best_fx:
                        best_fx = f_polish
                        best_x = x_polish.copy()
                    if sigma_final < 1e-14:
                        break

        return best_x, best_fx
