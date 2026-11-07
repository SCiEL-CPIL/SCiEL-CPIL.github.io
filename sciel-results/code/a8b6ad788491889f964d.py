"""Multi-start Nelder-Mead optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Best-first Nelder-Mead optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0

        # 1. Phase 1: Exploration
        # Generate 10 deterministic starting points
        n_starts = 10
        base_points = [
            [-3.0, -3.0, -3.0, -3.0, -3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [-3.0, 3.0, -3.0, 3.0, -3.0],
            [3.0, -3.0, 3.0, -3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 2.0, -2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0, -2.0, 2.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, 1.0],
            [-4.0, 4.0, -4.0, 4.0, -4.0],
        ]
        
        best_x = None
        best_f = np.inf
        
        # Evaluate all starts
        for p in base_points:
            if calls_made >= budget:
                break
            x = np.clip(np.array(p, dtype=float), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls_made >= budget or best_x is None:
            return best_x, best_f

        # 2. Phase 2: Exploitation with Nelder-Mead on the best point
        # Initialize simplex around best_x
        x0 = best_x
        simplex = [x0.copy()]
        fx_list = [best_f]
        
        # Initial step size for simplex
        step = 0.5
        for d in range(dim):
            if calls_made >= budget:
                break
            xp = x0.copy()
            xp[d] += step
            xp = np.clip(xp, -5, 5)
            simplex.append(xp)
            fx_list.append(func(xp))
            calls_made += 1
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Run NM until budget exhausted
        while calls_made < budget:
            # Sort simplex
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]
            
            # Check if we can proceed
            if calls_made >= budget:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            
            if calls_made >= budget:
                break
            
            if fr < fx_list[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                if calls_made >= budget:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fx_list[-1] = fe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fr
            elif fr < fx_list[-1]:
                # Accept reflection
                simplex[-1] = xr
                fx_list[-1] = fr
            else:
                # Contraction
                if fr < fx_list[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_made += 1
                if calls_made >= budget:
                    break
                
                if fc <= min(fr, fx_list[-1]):
                    simplex[-1] = xc
                    fx_list[-1] = fc
                else:
                    # Shrink
                    for j in range(1, dim + 1):
                        if calls_made >= budget:
                            break
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], -5, 5)
                        fx_list[j] = func(simplex[j])
                        calls_made += 1
                    if calls_made >= budget:
                        break
            
            # Update global best
            min_idx = np.argmin(fx_list)
            if fx_list[min_idx] < best_f:
                best_f = fx_list[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
