"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:
            return None
        fx = func(x)
        self._calls_made += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()
        return fx

    def _nelder_mead(self, func, x0, max_evals):
        """
        Nelder-Mead simplex method for local optimization.
        """
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        f0 = self._evaluate(func, x0)
        if f0 is None:
            return x0, self._best_f
        
        simplex_f = [f0]
        
        # Add vertices along axes with small step
        step = 0.05
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += step
            simplex.append(x_new)
            f_new = self._evaluate(func, x_new)
            if f_new is None:
                return x0, self._best_f
            simplex_f.append(f_new)
            
        # Sort simplex by function value
        def sort_simplex():
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            return simplex, simplex_f

        simplex, simplex_f = sort_simplex()
        
        # Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        tol = 1e-8
        
        for _ in range(max_evals):
            if self._calls_made >= self.budget or time.time() - self._start_time > 290:
                break
                
            # Check convergence
            if abs(simplex_f[0] - simplex_f[-1]) < tol:
                break
                
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = self._evaluate(func, x_ref)
            if f_ref is None:
                break
                
            if f_ref < simplex_f[0]:
                # Try expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = self._evaluate(func, x_exp)
                if f_exp is None:
                    break
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = self._evaluate(func, x_con)
                if f_con is None:
                    break
                
                if f_con < simplex_f[-1]:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = self._evaluate(func, simplex[i])
                        if simplex_f[i] is None:
                            return x0, self._best_f
            
            simplex, simplex_f = sort_simplex()

        return simplex[0], simplex_f[0]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf

        # Phase 1: (1+1)-ES for global search and early progress
        # Start at center
        x = np.zeros(dim)
        fx = self._evaluate(func, x)
        
        if fx is None:
            return self._best_x, self._best_f

        sigma = 2.5
        rng = np.random.default_rng(42)
        
        # Use 80% of budget for ES
        es_budget = int(budget * 0.8)
        
        for i in range(es_budget):
            if self._calls_made >= budget or time.time() - self._start_time > 290:
                break
            
            # Generate candidate
            x_candidate = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fx_candidate = self._evaluate(func, x_candidate)
            if fx_candidate is None:
                break
                
            if fx_candidate < fx:
                x = x_candidate
                fx = fx_candidate
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            # If sigma becomes very small, we might be close to optimum
            # But we keep going to use budget efficiently until switch to NM

        # Phase 2: Nelder-Mead for local refinement
        remaining = budget - self._calls_made
        if remaining > 10:
            x_nm, f_nm = self._nelder_mead(func, x, remaining)
            if f_nm < self._best_f:
                self._best_f = f_nm
                self._best_x = x_nm

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
