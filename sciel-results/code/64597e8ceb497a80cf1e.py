"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with adaptive restarts for Sphere minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # Initial simplex
        # Start at 0, which is a reasonable guess given the domain [-5, 5]^5
        # and that the minimizer is in [-4, 4]^5.
        x0 = np.zeros(dim)
        
        # Define initial simplex points
        # The first point is x0.
        # Subsequent points are x0 + delta * e_i
        # We choose delta to be small enough to stay in bounds but large enough to explore.
        # Since the domain is [-5, 5], a step of 0.5 is safe from 0.
        delta = 0.5
        
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0
            simplex[i + 1][i] += delta

        # Evaluate initial simplex
        values = np.zeros(dim + 1)
        calls_made = 0
        
        for i in range(dim + 1):
            if time.time() >= deadline or calls_made >= budget:
                break
            values[i] = func(simplex[i])
            calls_made += 1

        if calls_made == 0:
             return x0, float('inf')

        best_idx = np.argmin(values)
        best_x = simplex[best_idx].copy()
        best_f = values[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Track stagnation
        no_improve_count = 0
        max_no_improve = 20 * dim # Reset if stuck for a while

        while calls_made < budget and time.time() < deadline:
            # Sort simplex by function values
            sort_idx = np.argsort(values)
            simplex = simplex[sort_idx]
            values = values[sort_idx]
            
            # Update best
            if values[0] < best_f:
                best_f = values[0]
                best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for convergence or stagnation
            if no_improve_count > max_no_improve:
                # Restart with a smaller simplex around the best point
                # Or just shrink aggressively
                # Let's shrink the whole simplex around the best point
                center = best_x
                simplex = center + sigma * (simplex - center)
                # Re-evaluate? No, we can just continue, but we need new values.
                # To save calls, we might just continue with current values, but they are stale.
                # Better: re-evaluate the worst few points? Or just let the algorithm shrink.
                # Standard NM shrinks the worst points towards the best.
                # If we are stuck, let's force a shrink of all points except the best.
                for i in range(1, dim + 1):
                    simplex[i] = center + sigma * (simplex[i] - center)
                    # We don't have values for these new points. 
                    # We must evaluate them. This costs dim calls.
                    # If we are close to deadline, this is risky.
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    values[i] = func(simplex[i])
                    calls_made += 1
                    if values[i] < best_f:
                        best_f = values[i]
                        best_x = simplex[i].copy()
                        no_improve_count = 0
                no_improve_count = 0 # Reset counter after restart attempt
                continue

            centroid = np.mean(simplex[1:], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[0])
            if time.time() >= deadline or calls_made >= budget:
                break
            fr = func(np.clip(xr, -5, 5))
            calls_made += 1
            
            if fr < values[-1]:
                if fr < values[0]:
                    # Try Expansion
                    xe = centroid + gamma * (centroid - simplex[0])
                    if time.time() >= deadline or calls_made >= budget:
                        break
                    fe = func(np.clip(xe, -5, 5))
                    calls_made += 1
                    if fe < fr:
                        simplex[0] = xe
                        values[0] = fe
                    else:
                        simplex[0] = xr
                        values[0] = fr
                else:
                    simplex[0] = xr
                    values[0] = fr
            else:
                # Contraction
                if fr < values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    if time.time() >= deadline or calls_made >= budget:
                        break
                    fc = func(np.clip(xc, -5, 5))
                    calls_made += 1
                    if fc <= fr:
                        simplex[0] = xc
                        values[0] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() >= deadline or calls_made >= budget:
                                break
                            values[i] = func(simplex[i])
                            calls_made += 1
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - xr)
                    if time.time() >= deadline or calls_made >= budget:
                        break
                    fc = func(np.clip(xc, -5, 5))
                    calls_made += 1
                    if fc < values[-1]:
                        simplex[0] = xc
                        values[0] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() >= deadline or calls_made >= budget:
                                break
                            values[i] = func(simplex[i])
                            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
