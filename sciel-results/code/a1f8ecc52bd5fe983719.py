import numpy as np
import time
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls = 0
        best_x = None
        best_fx = np.inf
        
        # Phase 1: Multi-start with Sobol initialization
        n_restarts = 10
        local_budget = 30
        
        # Generate initial points
        if HAS_SCIPY:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            n_samples = 16
            points = sampler.random(n_samples)
            points = points * 10.0 - 5.0
        else:
            # Fallback to random
            points = np.random.uniform(-5, 5, (n_restarts, self.dim))

        for i in range(n_restarts):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            x0 = points[i]
            x_opt, f_opt, calls_used = self._nelder_mead(func, x0, local_budget, deadline)
            calls += calls_used
            
            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt.copy()
        
        # Phase 2: Final Polish
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining_budget = self.budget - calls
            x_opt, f_opt, calls_used = self._nelder_mead(func, best_x, remaining_budget, deadline)
            calls += calls_used
            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt.copy()
        
        # Fallback if no valid point found
        if best_x is None:
            x0 = np.random.uniform(-5, 5, self.dim)
            x_opt, f_opt, calls_used = self._nelder_mead(func, x0, self.budget - calls, deadline)
            best_x = x_opt
            best_fx = f_opt

        return best_x, best_fx

    def _nelder_mead(self, func, x0, budget, deadline):
        """
        Nelder-Mead simplex method.
        budget: max number of function evaluations allowed for this run.
        Returns (best_x, best_fx, calls_made)
        """
        dim = self.dim
        
        # If budget is too small for simplex init, just return x0
        if budget < dim + 1:
            fx = func(x0)
            return x0, fx, 1

        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        fvals = np.zeros(dim + 1)
        
        simplex[0] = x0
        fvals[0] = func(x0)
        calls_made = 1
        
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            step = 0.5
            simplex[i][i-1] += step
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals[i] = func(simplex[i])
            calls_made += 1
            
            if calls_made >= budget:
                # Out of budget during init
                best_idx = np.argmin(fvals)
                return simplex[best_idx], fvals[best_idx], calls_made
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = budget - calls_made
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
                
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Convergence check
            if np.abs(fvals[-1] - fvals[0]) < 1e-10:
                break
                
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        
        # Find best
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], calls_made
