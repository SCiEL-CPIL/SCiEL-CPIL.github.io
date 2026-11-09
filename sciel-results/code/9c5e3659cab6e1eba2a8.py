"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead for 5D Rosenbrock with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng, deadline):
        """Nelder-Mead simplex method with budget and deadline constraints."""
        n = self.dim
        # Create initial simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            xi = np.array(x0, dtype=float)
            xi[i] += 0.5
            simplex.append(xi)
        
        fx = [func(x) for x in simplex]
        calls = n + 1
        best_x = simplex[int(np.argmin(fx))]
        best_f = min(fx)
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        while calls < budget and time.time() < deadline:
            # Sort simplex by function value
            idx = np.argsort(fx)
            simplex = [simplex[i] for i in idx]
            fx = [fx[i] for i in idx]
            
            if fx[0] < best_f:
                best_f = fx[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fx[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < fx[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fx[-1] = fe
                else:
                    simplex[-1] = xr
                    fx[-1] = fr
            elif fr < worst_f:
                simplex[-1] = xr
                fx[-1] = fr
            else:
                # Contraction
                if fr < worst_f:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fx[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < worst_f:
                        simplex[-1] = xc
                        fx[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
            
            if calls >= budget:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Quick random search to find promising region
        # Use first 50 calls for exploration
        explore_budget = min(50, self.budget // 10)
        best_x = None
        best_f = np.inf
        
        for _ in range(explore_budget):
            if time.time() >= deadline:
                break
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        # Phase 2: Multi-restart Nelder-Mead
        remaining_budget = self.budget - explore_budget
        n_restarts = 5
        calls_per_restart = remaining_budget // n_restarts
        
        for restart in range(n_restarts):
            if time.time() >= deadline:
                break
            if remaining_budget <= 0:
                break
            
            # Perturb best point for diversity
            if restart == 0:
                x0 = best_x.copy()
            else:
                x0 = best_x + rng.normal(0, 0.5, self.dim)
                x0 = np.clip(x0, -4, 4)
            
            x_best, f_best, calls_used = self._nelder_mead(
                func, x0, calls_per_restart, rng, deadline
            )
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            remaining_budget -= calls_used
        
        return best_x, best_f
# EVOLVE-BLOCK-END
