"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex with adaptive restarts for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, budget_remaining, t0, deadline):
        """Run Nelder-Mead simplex optimization from x0."""
        n = self.dim
        # Initialize simplex: x0 and n points offset along each axis
        simplex = [x0.copy()]
        fx = [fx0]
        
        # Initial step size
        delta = 0.5
        for i in range(n):
            x_i = x0.copy()
            x_i[i] += delta
            x_i = np.clip(x_i, -5, 5)
            simplex.append(x_i)
            fx.append(func(x_i))
            budget_remaining -= 1
            if budget_remaining <= 0:
                break
        
        if budget_remaining <= 0:
            return x0, fx0, budget_remaining

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        max_iter = min(budget_remaining, 200)  # Cap iterations to avoid wasting budget
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if budget_remaining <= 0:
                break
            
            # Sort by function value
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]
            
            # Check convergence
            if abs(fx[0] - fx[-1]) < 1e-10:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            budget_remaining -= 1
            
            if fx[0] <= f_ref < fx[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fx[-1] = f_ref
            elif f_ref < fx[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                budget_remaining -= 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fx[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fx[-1] = f_ref
            else:
                # Contraction
                if f_ref < fx[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                budget_remaining -= 1
                
                if f_con <= min(f_ref, fx[-1]):
                    simplex[-1] = x_con
                    fx[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx[i] = func(simplex[i])
                        budget_remaining -= 1
                    if budget_remaining <= 0:
                        break
        
        # Find best point
        order = np.argsort(fx)
        best_idx = order[0]
        return simplex[best_idx], fx[best_idx], budget_remaining

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Random initialization - evaluate 10 diverse starting points
        # Include origin and some structured points
        candidates = []
        
        # Origin
        x_origin = np.zeros(n)
        fx_origin = func(x_origin)
        budget -= 1
        candidates.append((x_origin, fx_origin))
        
        # Random points
        n_random = min(9, budget - 1)
        for _ in range(n_random):
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            budget -= 1
            candidates.append((x, fx))
        
        if budget <= 0:
            best = min(candidates, key=lambda c: c[1])
            return best[0], best[1]
        
        # Sort candidates by function value
        candidates.sort(key=lambda c: c[1])
        
        best_x = candidates[0][0].copy()
        best_f = candidates[0][1]
        
        # Phase 2: Run Nelder-Mead from the best few starting points
        # Allocate remaining budget across multiple restarts
        n_restarts = min(3, len(candidates))
        budget_per_restart = max(1, budget // n_restarts)
        
        for i in range(n_restarts):
            if budget <= 0:
                break
            if time.time() > deadline:
                break
            
            start_x = candidates[i][0].copy()
            start_f = candidates[i][1]
            
            # Only run NM if this start is promising (not too far from best)
            if start_f > best_f + 1.0 and i > 0:
                continue
            
            x_best, f_best, budget = self._nelder_mead(
                func, start_x, start_f, budget, t0, deadline
            )
            
            if f_best < best_f:
                best_x = x_best
                best_f = f_best
        
        return best_x, best_f
# EVOLVE-BLOCK-END
