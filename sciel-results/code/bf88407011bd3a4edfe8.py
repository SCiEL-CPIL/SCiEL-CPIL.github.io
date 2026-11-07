"""CMA-ES based optimizer for BBOB f19 (Griewank-Rosenbrock).

CMA-ES with covariance adaptation to capture the rotated landscape structure.
Includes a final local polish pass with Nelder-Mead and L-BFGS-B.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        rng = np.random.default_rng(42)
        
        def elapsed():
            return time.time() - t0
        
        def sample(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            return x, f
        
        # Phase 1: CMA-ES
        # Initialize with a diverse starting point
        x0 = rng.uniform(-3, 3, self.dim)
        x0, f0 = sample(x0)
        
        best_x = x0.copy()
        best_f = f0
        
        # CMA-ES parameters
        sigma = 2.0  # initial step size
        pop_size = 4 + int(3 * np.log(self.dim))  # lambda
        pop_size = max(pop_size, 6)
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        # CMA-ES state
        C = np.eye(self.dim)  # covariance matrix
        C_inv = np.eye(self.dim)
        pc = np.zeros(self.dim)  # evolution path for covariance
        ps = np.zeros(self.dim)  # evolution path for step size
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = (2 / ((self.dim + 1.3) ** 2 + mu_eff)) * (1 - cc)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Track best
        best_x = x0.copy()
        best_f = f0
        calls_made = 1
        
        # CMA-ES loop
        gen = 0
        while elapsed() < deadline and calls_made < self.budget - 5:
            # Compute eigenvalues of C for step size adaptation
            try:
                eigvals = np.linalg.eigvalsh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                D_inv = 1.0 / D
            except Exception:
                break
            
            # Generate population
            offspring = []
            offspring_vals = []
            
            for i in range(pop_size):
                if elapsed() >= deadline or calls_made >= self.budget - 5:
                    break
                
                # Sample from N(0, C)
                z = rng.standard_normal(self.dim)
                # Apply C^{1/2} using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except Exception:
                    y = z
                
                xi = x0 + sigma * y
                xi, fi = sample(xi)
                calls_made += 1
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                
                offspring.append(xi)
                offspring_vals.append(fi)
            
            if len(offspring) < mu:
                break
            
            # Sort by fitness
            idx = np.argsort(offspring_vals)
            offspring = [offspring[i] for i in idx]
            offspring_vals = [offspring_vals[i] for i in idx]
            
            # Update step size
            # ps = (1-cs) * ps + sqrt(cs*(2-cs)*mu_eff) / sigma * C^{-1/2} * (x_{1:lambda} - x_old)
            y_mean = np.zeros(self.dim)
            for i in range(mu):
                y_mean += weights[i] * (offspring[i] - x0) / sigma
            
            # C^{-1/2} * y_mean
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_y = C_inv_sqrt @ y_mean
            except Exception:
                C_inv_sqrt_y = y_mean
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt_y
            
            # Update mean
            x0_new = x0.copy()
            for i in range(mu):
                x0_new += weights[i] * (offspring[i] - x0)
            
            # Update covariance
            delta_sig = np.linalg.norm(ps) / chi_n - 1
            hsig = (delta_sig < (1 + 2 / (self.dim + 1)) * (1 - cs)) / (1 - cs)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_mean
            
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C)
            #    + cmu * sum_i w_i * (y_i @ y_i.T)
            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                y_i = (offspring[i] - x0) / sigma
                C_new += cmu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure positive definite
            C = C_new
            # Symmetrize
            C = (C + C.T) / 2
            
            # Add small regularization if needed
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-10:
                    C += 1e-10 * np.eye(self.dim)
            except Exception:
                C += 1e-10 * np.eye(self.dim)
            
            x0 = x0_new
            gen += 1
            
            # Adapt sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.linalg.norm(x0 - best_x) < 1e-10:
                # Check if we're stuck
                pass
        
        # Phase 2: Local polish with Nelder-Mead
        if elapsed() < deadline - 5:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: L-BFGS-B polish
        if elapsed() < deadline - 2:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Final check
        if elapsed() < deadline - 0.5:
            try:
                _, f_final = sample(best_x)
                if f_final < best_f:
                    best_f = f_final
            except Exception:
                pass
        
        return best_x
# EVOLVE-BLOCK-END
