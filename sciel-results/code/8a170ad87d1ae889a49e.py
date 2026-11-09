import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        
        def safe_func(x):
            self.calls += 1
            # Check deadline: 300s limit, keep 10s margin
            if time.time() - self.start_time > 290:
                raise TimeoutError()
            
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # 1. Global Search: Sample 20 random points
        n_global = 20
        if self.budget < n_global:
             n_global = self.budget
             
        # Generate random points in [-5, 5]^dim
        # Using standard normal scaled and clipped is often better than uniform for finding peaks/valleys, 
        # but uniform is fine for a broad search in a box.
        # Let's use uniform for simplicity and coverage.
        points = np.random.uniform(-5, 5, (n_global, self.dim))
        
        for i in range(n_global):
            if self.calls >= self.budget:
                break
            x = points[i]
            try:
                safe_func(x)
            except TimeoutError:
                break

        if self.best_x is None:
            # Should not happen if budget > 0
            return self.best_x, self.best_f

        # 2. Local Search: Nelder-Mead
        # We have used n_global calls. Remaining budget: self.budget - self.calls
        # Nelder-Mead requires dim+1 points for the simplex.
        # Each iteration uses 1 to 4 function evaluations.
        
        x0 = self.best_x
        f0 = self.best_f
        
        # Initialize simplex
        # Start with a small step size. Since we don't know the scale, 
        # a step of 0.1 to 1.0 is reasonable. 
        # Bent Cigar has a steep dimension, so a small initial step helps avoid overshooting.
        step_size = 0.1
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += step_size
        
        # Evaluate initial simplex
        # Note: simplex[0] is already evaluated as f0
        simplex_f = np.zeros(self.dim + 1)
        simplex_f[0] = f0
        
        for i in range(1, self.dim + 1):
            if self.calls >= self.budget:
                break
            try:
                simplex_f[i] = safe_func(simplex[i])
            except TimeoutError:
                break

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Tolerances
        ftol = 1e-12
        xtol = 1e-10
        max_iter = 1000 # Internal max, we rely on budget/time
        
        for _ in range(max_iter):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > 290:
                break

            # Sort simplex by function value
            idx = np.argsort(simplex_f)
            simplex = simplex[idx]
            simplex_f = simplex_f[idx]
            
            # Check convergence
            if np.max(simplex_f) - np.min(simplex_f) < ftol:
                break
            if np.max(np.abs(simplex - simplex[0])) < xtol:
                break

            # Centroid of all points except the worst
            x_bar = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = x_bar + alpha * (x_bar - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if self.calls >= self.budget:
                break
            try:
                f_r = safe_func(x_r)
            except TimeoutError:
                break

            if f_r < simplex_f[0]:
                # 2. Expansion
                x_e = x_bar + gamma * (x_r - x_bar)
                x_e = np.clip(x_e, -5, 5)
                
                if self.calls >= self.budget:
                    # If we can't evaluate expansion, accept reflection if it's better
                    if f_r < simplex_f[-1]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    break
                try:
                    f_e = safe_func(x_e)
                except TimeoutError:
                    if f_r < simplex_f[-1]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # 3. Contraction
                # Outside contraction if f_r >= simplex_f[-1], else inside
                if f_r < simplex_f[-1]:
                    # Inside contraction
                    x_c = x_bar + rho * (x_r - x_bar)
                else:
                    # Outside contraction
                    x_c = x_bar - rho * (simplex[-1] - x_bar)
                
                x_c = np.clip(x_c, -5, 5)
                
                if self.calls >= self.budget:
                    break
                try:
                    f_c = safe_func(x_c)
                except TimeoutError:
                    break

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # 4. Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        if self.calls >= self.budget:
                            break
                        try:
                            simplex_f[i] = safe_func(simplex[i])
                        except TimeoutError:
                            break
                    if self.calls >= self.budget:
                        break

        return self.best_x, self.best_f
