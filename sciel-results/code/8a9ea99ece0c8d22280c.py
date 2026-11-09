"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Standard CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2)**1.7 + mu_eff)
        c_mu = min(1 - c_1, (2 + (mu_eff / dim - 1) * weights[0]) / ((dim + 2)**1.7 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = 1.0
        B = np.eye(dim)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial population
        pop = []
        for i in range(lam):
            z = np.random.standard_normal(dim)
            x_i = x_mean + sigma * z
            x_i = np.clip(x_i, -5, 5)
            pop.append(x_i)
            
        f_pop = np.array([eval_func(x) for x in pop])
        
        # Sort
        idx = np.argsort(f_pop)
        pop = [pop[i] for i in idx]
        f_pop = f_pop[idx]
        
        while call_count < budget and time.time() < deadline:
            # Update mean
            x_old = x_mean
            x_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
            
            # Update step size
            y_sigma = (x_mean - x_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**2) / chi_n < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            delta_heav = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + delta_heav - 1) * np.eye(dim)
            
            y_w = np.array([(x_mean - x_old) / sigma for _ in range(mu)]) # Wait, y_w should be weighted sum of (x_i - x_old)/sigma
            # Correct y_w calculation
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (pop[i] - x_old) / sigma
                
            C += c_mu * np.outer(y_w, y_w)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Decompose C
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-20, None)
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # Reset if decomposition fails
                C = np.eye(dim)
                D = 1.0
                B = np.eye(dim)
                eigvals = np.ones(dim)
                
            # Generate new population
            new_pop = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                new_pop.append(x_i)
                
            f_new_pop = np.array([eval_func(x) for x in new_pop])
            
            # Sort
            idx = np.argsort(f_new_pop)
            pop = [new_pop[i] for i in idx]
            f_pop = f_new_pop[idx]
            
        return best_x, best_f
# EVOLVE-BLOCK-END
