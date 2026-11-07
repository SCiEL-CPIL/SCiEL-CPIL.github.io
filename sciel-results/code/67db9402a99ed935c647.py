import numpy as np
import time

class CMAES:
    def __init__(self, dim, x0, sigma0, rng):
        self.dim = dim
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.rng = rng
        self.lambda_ = max(4 * dim, 10)
        self.mu = self.lambda_ // 2
        self.weights = np.zeros(self.lambda_)
        w = np.log(self.lambda_ / 2 + 1) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights[:self.mu] = w[:self.mu]
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights[:self.mu] ** 2)
        
        n = dim
        self.c_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.lambda_ + 1)) - 1) / (n + self.mu_eff)
        self.c_c = 4 / (n + 4)
        self.c_1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(self.c_1 / 2, self.mu_eff / 2 - self.c_1 / 2)
        
        self.best_x = self.x.copy()
        self.best_f = float('inf')
        self.stagnation = 0
        self.eval_count = 0

    def step(self, func, budget_remaining):
        if self.eval_count + self.lambda_ > budget_remaining:
            # Only take as many as available, or just 1 if very tight
            # For simplicity, if we can't afford a full pop, we stop this instance
            # Or we can reduce lambda. Let's just check if we have at least 1.
            if self.eval_count >= budget_remaining:
                return False
        
        # Generate offspring
        y = np.zeros((self.lambda_, self.dim))
        for i in range(self.lambda_):
            # Standard normal
            z = self.rng.standard_normal(self.dim)
            # Transform using Cholesky of C
            # We need to update C's Cholesky factor occasionally or just use eigendecomp
            # For small dim, eigendecomp is fine every step
            eigvals, eigvecs = np.linalg.eigh(self.C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-10)
            y[i] = self.x + self.sigma * (eigvecs @ (np.sqrt(eigvals) * z))
            y[i] = np.clip(y[i], -5, 5)
            
            self.eval_count += 1
            fy = func(y[i])
            if fy < self.best_f:
                self.best_f = fy
                self.best_x = y[i].copy()
                
        # Sort offspring
        f_vals = np.array([func(y[i]) for i in range(self.lambda_)]) # Re-evaluating is bad, we should store
        # Correction: We must store function values during generation to avoid re-evaluation.
        # Let's rewrite the generation loop to store values.

    def step_optimized(self, func, budget_remaining):
        if self.eval_count + self.lambda_ > budget_remaining:
            return False
            
        y = np.zeros((self.lambda_, self.dim))
        f_vals = np.zeros(self.lambda_)
        
        # Precompute Cholesky/Eig
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        for i in range(self.lambda_):
            z = self.rng.standard_normal(self.dim)
            y[i] = self.x + self.sigma * (L @ z)
            y[i] = np.clip(y[i], -5, 5)
            
            self.eval_count += 1
            f_vals[i] = func(y[i])
            if f_vals[i] < self.best_f:
                self.best_f = f_vals[i]
                self.best_x = y[i].copy()

        # Sort
        idx = np.argsort(f_vals)
        y_sorted = y[idx]
        f_sorted = f_vals[idx]
        
        # Update best
        # (Already done in loop)
        
        # Check improvement for stagnation
        if f_sorted[0] < self.best_f_prev:
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.best_f_prev = f_sorted[0]
        
        # Update mean
        self.x_new = np.sum(self.weights[:self.mu, np.newaxis] * y_sorted[:self.mu], axis=0)
        
        # Update step size
        y_sigma = (self.x_new - self.x) / self.sigma
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_sigma
        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.ps) / self.chi_n - 1))
        
        # Update covariance
        y_diff = (self.x_new - self.x) / self.sigma
        self.pc = (1 - self.c_c) * self.pc + self.c_c * y_diff
        
        # Rank one update
        C_rank1 = np.outer(y_diff, y_diff)
        
        # Rank mu update
        C_rankmu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y_i = (y_sorted[i] - self.x) / self.sigma
            C_rankmu += self.weights[i] * np.outer(y_i, y_i)
            
        self.C = (1 + self.c_1 * (self.mu_eff - 2 + 2 / self.c_c) - self.c_mu) * self.C \
                 + self.c_1 * C_rank1 \
                 + self.c_mu * C_rankmu
                 
        # Symmetrize and fix
        self.C = (self.C + self.C.T) / 2
        # Ensure PD
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        self.C = (eigvecs * eigvals) @ eigvecs.T
        
        self.x = self.x_new
        return True

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        
        # Initialize multiple instances
        n_instances = 5
        instances = []
        # LHS for initial points
        x_starts = np.random.RandomState(42).uniform(-5, 5, (n_instances, self.dim))
        for i in range(n_instances):
            inst = CMAES(self.dim, x_starts[i], 2.0, rng)
            # Initial evaluation of center
            fx = func(inst.x)
            inst.eval_count += 1
            inst.best_f = fx
            inst.best_x = inst.x.copy()
            inst.best_f_prev = fx
            instances.append(inst)
            
        total_evals = n_instances
        best_f_global = min(inst.best_f for inst in instances)
        best_x_global = min(instances, key=lambda x: x.best_f).best_x.copy()
        
        while total_evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select best instance
            best_idx = np.argmin([inst.best_f for inst in instances])
            current_inst = instances[best_idx]
            
            # Check if we can afford a step
            if current_inst.eval_count + current_inst.lambda_ > self.budget - total_evals:
                # Try to fit what we can? CMA-ES needs full pop.
                # If we can't afford a full pop, we might switch to a different strategy or just stop.
                # Let's see if we can run a smaller pop? No, standard CMA-ES is fixed.
                # If we are close to budget, we might just do random restarts or stop.
                # For now, if we can't do a full step, break.
                break
                
            # Check stagnation
            if current_inst.stagnation > 20:
                # Restart this instance
                new_x = rng.uniform(-5, 5, self.dim)
                current_inst.x = new_x
                current_inst.C = np.eye(self.dim)
                current_inst.pc = np.zeros(self.dim)
                current_inst.ps = np.zeros(self.dim)
                current_inst.sigma = 2.0
                current_inst.stagnation = 0
                current_inst.best_f = float('inf')
                current_inst.best_f_prev = float('inf')
                # Re-evaluate center
                fx = func(new_x)
                current_inst.eval_count += 1
                total_evals += 1
                current_inst.best_f = fx
                current_inst.best_x = new_x.copy()
                current_inst.best_f_prev = fx
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = new_x.copy()
                continue

            # Step
            success = current_inst.step_optimized(func, self.budget - total_evals)
            if not success:
                break
                
            total_evals += current_inst.lambda_
            
            # Update global best
            if current_inst.best_f < best_f_global:
                best_f_global = current_inst.best_f
                best_x_global = current_inst.best_x.copy()

        return best_x_global, best_f_global
