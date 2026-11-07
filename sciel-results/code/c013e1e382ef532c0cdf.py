"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Wrap func to count calls and enforce deadline
        call_count = 0
        def safe_call(x):
            nonlocal call_count
            if call_count >= budget or time.time() > deadline:
                return float('inf')
            call_count += 1
            return func(x)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = safe_call(x)
        best_x = x.copy()
        best_fx = fx

        # Nelder-Mead parameters
        # Initial simplex size
        delta = 0.5
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        # Stagnation detection
        stagnation_counter = 0
        max_stagnation = 20
        prev_best_fx = fx

        while call_count < budget and time.time() < deadline:
            # Initialize simplex if first iteration or restart
            if not hasattr(self, '_simplex') or self._simplex is None:
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x
                for i in range(1, dim + 1):
                    simplex[i] = x.copy()
                    simplex[i, i - 1] += delta
                    # Ensure within bounds
                    simplex[i] = np.clip(simplex[i], -5, 5)
                self._simplex = simplex
                simplex_f = np.array([safe_call(s) for s in simplex])
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_fx:
                    best_fx = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
                    prev_best_fx = best_fx
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                simplex = self._simplex
                simplex_f = np.array([safe_call(s) for s in simplex])
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_fx:
                    best_fx = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
                    if best_fx < prev_best_fx - 1e-8:
                        stagnation_counter = 0
                        prev_best_fx = best_fx
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Check for stagnation and restart
            if stagnation_counter > max_stagnation:
                # Restart strategy
                # 50% chance to perturb current best, 50% chance to go random
                if rng.random() < 0.5:
                    x = best_x + rng.normal(0, 1, dim) * 0.5
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                self._simplex = None # Force re-initialization of simplex
                delta = 1.0 # Reset delta for new simplex
                continue

            # Standard Nelder-Mead step
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = safe_call(simplex[i])
            
            self._simplex = simplex

        return best_x, best_fx
# EVOLVE-BLOCK-END
