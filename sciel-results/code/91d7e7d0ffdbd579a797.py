"""Stochastic quasi-Newton optimizer with random restarts for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Nelder-Mead optimizer with random restarts.
    
    Strategy:
    - Use multiple random restarts to handle the hidden shift/rotation.
    - Within each restart, use Nelder-Mead (derivative-free) to optimize.
    - Nelder-Mead is much more budget-efficient than finite-difference L-BFGS.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320

    def _nelder_mead(self, func, x0, max_iter=100, tol=1e-6):
        """
        Nelder-Mead optimization.
        Returns (x_best, fx_best, calls_used).
        """
        dim = self.dim
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.5  # Initial step size
            
        # Evaluate initial simplex
        fx = np.array([func(x) for x in simplex])
        calls = dim + 1
        
        # Sort vertices by function value
        order = np.argsort(fx)
        simplex = simplex[order]
        fx = fx[order]
        
        best_x = simplex[0].copy()
        best_fx = fx[0]
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        for it in range(max_iter):
            if time.time() > self.deadline:
                break
            if calls >= self.budget:
                break
                
            # Check convergence
            if np.std(fx) < tol:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            fx_r = func(x_r)
            calls += 1
            
            if fx_r < fx[0]:
                # Check for expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                fx_e = func(x_e)
                calls += 1
                
                if fx_e < fx_r:
                    simplex[-1] = x_e
                    fx[-1] = fx_e
                else:
                    simplex[-1] = x_r
                    fx[-1] = fx_r
            elif fx_r < fx[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fx[-1] = fx_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                fx_c = func(x_c)
                calls += 1
                
                if fx_c < fx[-1]:
                    simplex[-1] = x_c
                    fx[-1] = fx_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx[i] = func(simplex[i])
                        calls += 1
            
            # Re-sort
            order = np.argsort(fx)
            simplex = simplex[order]
            fx = fx[order]
            
            if fx[0] < best_fx:
                best_fx = fx[0]
                best_x = simplex[0].copy()
                
        return best_x, best_fx, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_used = 0
        
        # Budget allocation
        # Nelder-Mead uses ~1-2 calls per iter. 100 iters -> ~200 calls.
        # We can afford 2-3 restarts.
        n_restarts = 3
        max_iter_per_restart = 80
        
        for restart in range(n_restarts):
            if time.time() > self.deadline:
                break
            if calls_used >= self.budget:
                break
                
            # Random start
            x0 = rng.uniform(-4, 4, self.dim)
            fx0 = func(x0)
            calls_used += 1
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
                
            # Run Nelder-Mead
            x_opt, fx_opt, phase_calls = self._nelder_mead(func, x0, max_iter=max_iter_per_restart)
            calls_used += phase_calls
            
            if fx_opt < best_fx:
                best_fx = fx_opt
                best_x = x_opt.copy()
                
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
