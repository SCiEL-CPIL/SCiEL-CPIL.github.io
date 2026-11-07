# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    CMA-ES with adaptive restarts + Nelder-Mead polish for BBOB f7 (Step Ellipsoidal).
    
    Strategy:
    1. Multiple CMA-ES restarts from diverse starting points to handle the flat half-space.
    2. Nelder-Mead local polish on the best point found to exploit the smooth convex basin.
    3. Time-aware budget management to stay within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def _count_call(self):
        self._call_count += 1

    def _cma_es(self, func, x0, sigma0, max_calls, deadline, rng):
        """
        Run CMA-ES from starting point x0 with initial step size sigma0.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        calls_used = 0
        best_x = x.copy()
        best_f = np.inf
        no_improve = 0
        max_no_improve = 200  # stagnation threshold for restart
        
        # Initial evaluation
        fx = func(x)
        self._count_call()
        calls_used += 1
        best_f = fx
        best_x = x.copy()
        
        while calls_used < max_calls and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                offspring.append((xi, z, y))
            
            # Evaluate offspring
            f_offspring = []
            for xi, z, y in offspring:
                if calls_used >= max_calls or time.time() > deadline:
                    break
                fi = func(xi)
                self._count_call()
                calls_used += 1
                f_offspring.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            
            if len(f_offspring) < lambda_:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(f_offspring)
            offspring_sorted = [offspring[i] for i in sorted_idx[:mu]]
            f_sorted = [f_offspring[i] for i in sorted_idx[:mu]]
            
            # Check for stagnation
            if no_improve > max_no_improve:
                break
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * offspring_sorted[i][2]
            x = x + sigma * y_w
            x = np.clip(x, -5, 5)
            
            # Update step size control
            y_sigma = C_inv_sqrt @ y_w
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used + 1) / (dim + 1))) / sigma < \
               (1 + 1 / d_sigma) * np.sqrt((mu_eff + 1) / (dim + 1)) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt((mu_eff + 1) / (dim + 1)) - 1))
            
            # Update covariance matrix
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used + 1) / (dim + 1))) < \
                (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) * (1 - c_sigma) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (np.outer(y_w, y_w) + (1 - h_sigma) * c_c * (2 - c_c) * C - np.outer(p_c, p_c))
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            # Update matrix square root (Cholesky-like)
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # Fallback: use eigendecomposition
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 50.0)
        
        return best_x, best_f, calls_used

    def _nelder_mead(self, func, x0, max_calls, deadline, rng):
        """
        Run Nelder-Mead local optimization from x0.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        calls_used = 0
        
        # Create initial simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(dim):
            perturbation = x0.copy()
            perturbation[i] += 0.5 if x0[i] < 0 else -0.5
            perturbation = np.clip(perturbation, -5, 5)
            simplex.append(perturbation)
        
        # Evaluate initial simplex
        f_simplex = []
        for i, xi in enumerate(simplex):
            if calls_used >= max_calls or time.time() > deadline:
                break
            fi = func(xi)
            self._count_call()
            calls_used += 1
            f_simplex.append(fi)
        
        if len(f_simplex) < dim + 1:
            best_idx = np.argmin(f_simplex)
            return simplex[best_idx], f_simplex[best_idx], calls_used
        
        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5 # shrinkage
        max_iter = 200
        
        for _ in range(max_iter):
            if calls_used >= max_calls or time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            if calls_used >= max_calls or time.time() > deadline:
                break
            f_r = func(x_r)
            self._count_call()
            calls_used += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if calls_used >= max_calls or time.time() > deadline:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    break
                f_e = func(x_e)
                self._count_call()
                calls_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                if calls_used >= max_calls or time.time() > deadline:
                    break
                f_c = func(x_c)
                self._count_call()
                calls_used += 1
                if f_c < min(f_r, f_simplex[-1]):
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_used >= max_calls or time.time() > deadline:
                            break
                        f_simplex[i] = func(simplex[i])
                        self._count_call()
                        calls_used += 1
        
        # Find best in simplex
        best_idx = np.argmin(f_simplex)
        return simplex[best_idx], f_simplex[best_idx], calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Wrap func to count calls
        def wrapped(x):
            self._count_call()
            return func(x)
        
        # Phase 1: Initial diverse exploration
        n_initial = 10
        initial_points = []
        
        # Center
        initial_points.append(np.zeros(self.dim))
        # Corners of a smaller box
        for i in range(1, n_initial):
            x = rng.uniform(-3, 3, self.dim)
            initial_points.append(x)
        
        # Evaluate initial points
        for x in initial_points:
            if self._call_count >= self.budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = wrapped(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
        
        # Phase 2: CMA-ES restarts from diverse starting points
        n_restarts = 5
        budget_per_restart = max(30, (self.budget - self._call_count - 50) // n_restarts)
        
        for r in range(n_restarts):
            if self._call_count >= self.budget or time.time() > deadline:
                break
            
            # Choose starting point: mix of best found and random
            if r == 0:
                x0 = self._best_x.copy() if self._best_x is not None else np.zeros(self.dim)
                sigma0 = 1.0
            elif r == 1:
                x0 = self._best_x.copy() if self._best_x is not None else np.zeros(self.dim)
                x0 += rng.standard_normal(self.dim) * 0.5
                sigma0 = 0.5
            else:
                # Diverse random start
                x0 = rng.uniform(-4, 4, self.dim)
                sigma0 = 1.5
            
            x0 = np.clip(x0, -5, 5)
            
            # Run CMA-ES
            max_calls_for_cma = min(budget_per_restart, self.budget - self._call_count - 30)
            if max_calls_for_cma < 10:
                break
            
            best_x_cma, best_f_cma, calls_used = self._cma_es(
                wrapped, x0, sigma0, max_calls_for_cma, deadline, rng
            )
            
            if best_f_cma < self._best_f:
                self._best_f = best_f_cma
                self._best_x = best_x_cma.copy()
        
        # Phase 3: Nelder-Mead polish on the best point
        if self._best_x is not None and self._call_count < self.budget and time.time() < deadline:
            max_calls_nm = min(100, self.budget - self._call_count)
            if max_calls_nm > 5:
                best_x_nm, best_f_nm, calls_used = self._nelder_mead(
                    wrapped, self._best_x, max_calls_nm, deadline, rng
                )
                if best_f_nm < self._best_f:
                    self._best_f = best_f_nm
                    self._best_x = best_x_nm.copy()
        
        # Phase 4: Final (1+1)-ES polish if budget remains
        if self._best_x is not None and self._call_count < self.budget and time.time() < deadline:
            x = self._best_x.copy()
            fx = self._best_f
            sigma = 0.1
            
            while self._call_count < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < self._best_f:
                        self._best_f = fx
                        self._best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
