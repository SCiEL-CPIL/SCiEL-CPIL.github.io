import numpy as np
import time

class CMAES:
    def __init__(self, dim, rng, x0, sigma0, popsize=None):
        self.dim = dim
        self.rng = rng
        self.x = x0.copy()
        self.sigma = sigma0
        if popsize is None:
            self.lam = 4 + int(3 * np.log(dim))
        else:
            self.lam = popsize
        self.mu = self.lam // 2
        # Weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        
        # Parameters
        self.c_sigma = (self.mueff + 2) / (dim + 4)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mueff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2)**2 + self.mueff))
        
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.inv_C = np.eye(dim)
        self.chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        self.generation = 0
        self.best_x = x0.copy()
        self.best_f = np.inf
        self.no_improvement = 0

    def _update(self, x_sorted, f_sorted):
        self.generation += 1
        
        # Update mean
        old_x = self.x.copy()
        self.x = self.x + np.sum(self.weights[:, None] * (x_sorted[:self.mu] - self.x), axis=0)
        
        # Check for improvement
        if f_sorted[0] < self.best_f:
            self.best_f = f_sorted[0]
            self.best_x = x_sorted[0].copy()
            self.no_improvement = 0
        else:
            self.no_improvement += 1

        # Update evolution paths
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * self._transform(self.inv_C, (self.x - old_x) / self.sigma)
        self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * self._transform(self.inv_C, (self.x - old_x) / self.sigma)
        
        # Update step size
        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # Update covariance matrix
        delta_diag = (1 - 2 * self.c_1 + self.c_mu) * self.c_c * (2 - self.c_c)
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.sum(self.weights[:, None, None] * np.outer(x_sorted[:self.mu] - old_x, x_sorted[:self.mu] - old_x), axis=0) / (self.sigma**2)
        
        self.C = (1 + self.c_1 * delta_diag - self.c_1 - self.c_mu) * self.C + self.c_1 * rank_one + self.c_mu * rank_mu
        
        # Ensure symmetry and positive definiteness (approximate)
        self.C = (self.C + self.C.T) / 2
        # Regularize if needed
        eigvals, eigvecs = np.linalg.eigh(self.C)
        min_eig = np.min(eigvals)
        if min_eig < 1e-12:
            self.C += (1e-12 - min_eig) * np.eye(self.dim)
        
        self.inv_C = np.linalg.inv(self.C)

    def _transform(self, inv_C, v):
        # Efficiently compute C^{-1/2} v using Cholesky of C
        # Note: For small dim, direct inverse is fine, but let's stick to standard update
        # Actually, the standard CMA-ES update uses C^{-1/2} for the path updates.
        # For simplicity and stability in small dim, we can use the inverse directly if we are careful,
        # but the standard formula is:
        # ps = (1-cs)ps + sqrt(cs(2-cs) mueff) * C^{-1/2} (x - x_old) / sigma
        # Since we don't maintain C^{-1/2} explicitly, and dim is 5, we can compute it.
        L = np.linalg.cholesky(self.C)
        # C^{-1/2} v = L^{-T} L^{-1} v
        tmp = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, tmp)

    def step(self, func):
        # Sample
        L = np.linalg.cholesky(self.C)
        y = self.rng.standard_normal((self.lam, self.dim))
        y = np.einsum('ij,jk->ik', y, L.T) # y * L.T gives samples from N(0, C)
        x_pop = self.x + self.sigma * y
        
        # Evaluate
        f_pop = np.array([func(x) for x in x_pop])
        
        # Sort
        idx = np.argsort(f_pop)
        x_sorted = x_pop[idx]
        f_sorted = f_pop[idx]
        
        self._update(x_sorted, f_sorted)
        
        return self.best_f, self.best_x

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        time_limit = 290.0  # Safety margin
        
        best_x = None
        best_f = np.inf
        
        # Initial random points to find a decent start
        # Use 10% of budget for initialization
        init_budget = max(10, self.budget // 10)
        
        for i in range(init_budget):
            if self.calls >= self.budget:
                break
            if time.time() - start_time > time_limit:
                break
            
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            self.calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(self.dim)
            best_f = func(x)
            self.calls += 1
            best_x = x
        
        # Run CMA-ES with restarts
        # We have remaining budget
        remaining = self.budget - self.calls
        
        # Restart strategy:
        # Run CMA-ES for a fixed number of generations or until budget/time runs out.
        # If stagnation is detected, restart from the current best point with a larger sigma.
        
        sigma_init = 2.0
        stagnation_limit = 20
        
        while self.calls < self.budget:
            if time.time() - start_time > time_limit:
                break
                
            # Create CMA-ES instance
            # Start from current best, with initial sigma
            # If we've restarted before, maybe use a different start?
            # For simplicity, restart from best_x with a larger sigma to escape local min.
            
            cma = CMAES(self.dim, rng, best_x, sigma_init)
            
            # Run CMA-ES
            gen = 0
            while self.calls < self.budget:
                if time.time() - start_time > time_limit:
                    break
                
                # Check if we can do a full population step
                # CMA-ES uses self.lam calls per generation
                if self.calls + cma.lam > self.budget:
                    # Do partial step? Or just stop?
                    # Standard CMA-ES expects full population.
                    # If we have fewer calls than lam, we can't do a proper CMA-ES step.
                    # We can just do random local search or stop.
                    # Given dim=5, lam~13. If remaining < 13, we just do random perturbations.
                    break
                
                try:
                    f_best, x_best = cma.step(func)
                except Exception:
                    # If CMA-ES fails (e.g. numerical issues), break
                    break
                
                self.calls += cma.lam
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
                
                # Check stagnation
                if cma.no_improvement >= stagnation_limit:
                    # Stagnation detected, restart with larger sigma
                    sigma_init *= 1.5
                    break
                
                # Check if sigma is too small
                if cma.sigma < 1e-6:
                    break
                
                gen += 1
        
        return best_x, best_f

