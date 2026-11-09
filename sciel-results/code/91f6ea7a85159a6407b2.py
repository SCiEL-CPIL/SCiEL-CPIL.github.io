"""Deterministic multi-scale search with Nelder-Mead local refinement for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Deterministic multi-scale search with robust local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        calls = [0]
        dim = self.dim
        budget = self.budget

        def f(x):
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            # Clip to box [-5, 5] as per problem statement
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Phase 1: Coarse exploration using deterministic diverse points
        # Lunacek has two basins. We need to sample both.
        # Using a Halton-like sequence or fixed diverse points.
        # Let's use a fixed set of points that cover the space well.
        
        # Generate Halton sequence for deterministic low-discrepancy sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = [halton(i, base) for base in range(2, dim + 2)]
                # Scale to [-5, 5]
                point = [2 * 10 * p - 5 for p in point]
                points.append(np.array(point))
            return points

        # Evaluate a set of Halton points
        num_halton = 20
        halton_points = halton_sequence(num_halton, dim)
        
        # Also add some specific points likely to be near the minima
        # Global min is often near 0, local min is offset.
        # The offset in standard Lunacek is related to lambda.
        # We don't know the shift, so we rely on the Halton points to cover the space.
        
        candidate_starts = []
        for p in halton_points:
            candidate_starts.append(p)
            
        # Add origin and a few other strategic points
        candidate_starts.append(np.zeros(dim))
        candidate_starts.append(np.full(dim, 1.0))
        candidate_starts.append(np.full(dim, -1.0))
        
        # Evaluate all candidate starts
        evaluated_starts = []
        for x0 in candidate_starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = f(x0)
            update_best(x0, fx)
            evaluated_starts.append((x0, fx))

        # Phase 2: Local refinement using Nelder-Mead from the best point
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            # Use Nelder-Mead for local optimization
            # maxfev is the maximum number of function evaluations
            remaining_calls = budget - calls[0]
            # Reserve some calls for potential second-best optimization
            maxfev_nm = min(remaining_calls // 2, 200)
            
            try:
                res = minimize(f, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'maxfev': maxfev_nm, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: If budget remains, try local optimization from the second-best point
        # This helps if the first best was a local minimum and the second was in a better basin
        if calls[0] < budget and time.time() < deadline:
            # Sort evaluated starts by fitness and try the top few that are far from best_x
            evaluated_starts.sort(key=lambda t: t[1])
            for x0, fx0 in evaluated_starts[:5]:
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                if np.linalg.norm(x0 - best_x) < 0.5:
                    continue
                
                # Optimize from this point
                remaining_calls = budget - calls[0]
                maxfev_nm = min(remaining_calls // 2, 100)
                if maxfev_nm < 20:
                    break
                try:
                    res = minimize(f, x0, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'maxfev': maxfev_nm, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # Phase 4: Final polish with coordinate descent for high precision
        if calls[0] < budget and time.time() < deadline and best_x is not None:
            current_x = best_x.copy()
            current_f = best_f
            # Very small steps
            fine_steps = [1e-3, 1e-4, 1e-5, 1e-6]
            for sigma in fine_steps:
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = True
                while improved and calls[0] < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls[0] >= budget:
                            break
                        for sign in [1, -1]:
                            if calls[0] >= budget:
                                break
                            new_x = current_x.copy()
                            new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)
                            if np.allclose(new_x, current_x, atol=1e-12):
                                continue
                            fx = f(new_x)
                            if fx < current_f:
                                current_x = new_x
                                current_f = fx
                                update_best(current_x, current_f)
                                improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END
