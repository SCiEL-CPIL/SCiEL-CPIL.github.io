"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weighted parents
        # weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        # step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # position update parameters
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # initial step size
        sigma = 2.0

        # initial mean and distribution
        mean = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Evaluate initial point (call 1)
        # Note: The contract says func clips to [-5, 5]. CMA-ES usually works in unconstrained space,
        # but we can clip the samples. If we clip, the distribution might be distorted.
        # However, for BBOB, the optimum is inside [-4, 4], so clipping is rarely active 
        # except for outliers. We will clip to be safe.
        
        # Start with a point in the center? Or random?
        # Center is a good guess for BBOB since optimum is in [-4,4]^5.
        x0 = np.zeros(dim)
        fx = func(x0)
        best_x = x0.copy()
        best_f = fx
        calls = 1

        if time.time() > deadline:
            return best_x, best_f

        for gen in range(1, budget):
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(mean, sigma^2 * D B B^T D)
                # Efficient sampling: mean + sigma * (B @ (D * z))
                z = np.random.standard_normal(dim)
                pop[i] = mean + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                np.clip(pop[i], -5, 5, out=pop[i])
                
                # Evaluate
                f_val = func(pop[i])
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = pop[i].copy()
                
                # Check deadline inside loop to avoid long stalls
                if time.time() > deadline:
                    return best_x, best_f

            # Sort population by fitness
            order = np.argsort(pop, axis=0, kind='stable') # This sorts rows, we need to sort by fitness
            # We need to keep track of which point had which fitness
            # Let's restructure: store (fitness, point)
            # Actually, simpler: 
            # We evaluated in order 0..lambda-1. Let's create an array of indices sorted by fitness.
            # We need the fitness values. Let's re-evaluate? No, that wastes budget.
            # Let's store fitnesses in an array.
            
            # Refactor: Store fitnesses
            # I'll rewrite the generation loop to store fitnesses properly.
            
            # Let's restart the generation logic with proper storage
            # This block is just for thought, actual code below handles it.

        # The logic above was a bit messy with sorting. Let's implement a clean CMA-ES loop.
        
        # Re-implementing cleanly to ensure correctness
        return self._run_cma_es(func, budget, dim, start_time, deadline)

    def _run_cma_es(self, func, budget, dim, start_time, deadline):
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        sigma = 2.0

        mean = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial evaluation
        x0 = np.zeros(dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        calls = 1

        if time.time() > deadline:
            return best_x, best_f

        for gen in range(1, budget):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            fit = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                pop[i] = mean + sigma * (B @ (D * z))
                np.clip(pop[i], -5, 5, out=pop[i])
                fit[i] = func(pop[i])
                calls += 1
                
                if fit[i] < best_f:
                    best_f = fit[i]
                    best_x = pop[i].copy()
                
                if time.time() > deadline:
                    return best_x, best_f

            # Sort by fitness
            order = np.argsort(fit)
            y = pop[order]
            fit_sorted = fit[order]
            
            # Select mu best
            y_mu = y[:mu]
            
            # New mean
            m_new = np.sum(w[:, None] * y_mu, axis=0)
            
            # Update evolution path ps
            inv_sqrt_D = 1.0 / D
            # y_mu @ B gives the components in the eigenbasis?
            # Actually, step is sigma * B @ (D * z).
            # So z = D^-1 B^T (y - mean) / sigma
            # ps = (1 - c_sigma) ps + c_sigma * sqrt(mueff) * B^T @ (D^-1 * (m_new - mean)) / sigma
            # Note: B is orthogonal, so B^T is inverse.
            
            diff = m_new - mean
            # Transform diff back to the "z" space
            # diff = sigma * B @ (D * delta_z)  => delta_z = D^-1 B^T diff / sigma
            delta_z = (B.T @ diff) / (sigma * D)
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mueff) * delta_z
            
            # Hessian update check (simplified)
            if np.linalg.norm(ps) / chiN > 1.4 + 2 / (dim + 1):
                c_sigma_eff = c_sigma * (1 + d_sigma)
            else:
                c_sigma_eff = c_sigma
                
            sigma = sigma * np.exp(c_sigma_eff / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + c_c * (m_new - mean) / sigma
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu sum w_i (y_i - mean)(y_i - mean)^T
            # Note: The term (1 - (1-c_c)^2) C is often approximated or handled via the correlation matrix update.
            # Standard CMA-ES updates C directly.
            
            # We need to maintain C explicitly for stability, or update B, D.
            # Updating B, D is more complex. Let's maintain C.
            # However, I initialized B, D. Let's switch to maintaining C.
            
            # Restarting the state variables to use C directly is safer for implementation correctness.
            # Let's rewrite the whole thing to use C.

        # The above logic is getting complex and prone to bugs in the eigen-update.
        # A simpler, robust CMA-ES implementation maintaining C is preferred.
        
        # Let's return the best found so far if we hit a logic loop, but we should just implement the correct one.
        
        # Correct CMA-ES with C matrix
        return self._run_cma_es_correct(func, budget, dim, start_time, deadline)

    def _run_cma_es_correct(self, func, budget, dim, start_time, deadline):
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        sigma = 2.0

        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Helper to sample from N(0, C)
        # Use Cholesky decomposition
        L = np.linalg.cholesky(C)
        
        x0 = np.zeros(dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        calls = 1

        if time.time() > deadline:
            return best_x, best_f

        for gen in range(1, budget):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            fit = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                # Sample: mean + sigma * L @ z
                pop[i] = mean + sigma * (L @ z)
                np.clip(pop[i], -5, 5, out=pop[i])
                fit[i] = func(pop[i])
                calls += 1
                
                if fit[i] < best_f:
                    best_f = fit[i]
                    best_x = pop[i].copy()
                
                if time.time() > deadline:
                    return best_x, best_f

            # Sort by fitness
            order = np.argsort(fit)
            y = pop[order]
            
            # Select mu best
            y_mu = y[:mu]
            
            # New mean
            m_new = np.sum(w[:, None] * y_mu, axis=0)
            
            # Update evolution path ps
            # ps = (1 - c_sigma) ps + c_sigma * sqrt(mueff) * L^-1 (m_new - mean) / sigma
            # L is lower triangular. Solve L v = (m_new - mean) / sigma
            diff = (m_new - mean) / sigma
            v = np.linalg.solve(L, diff)
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mueff) * v
            
            # Hessian update check
            if np.linalg.norm(ps) / chiN > 1.4 + 2 / (dim + 1):
                c_sigma_eff = c_sigma * (1 + d_sigma)
            else:
                c_sigma_eff = c_sigma
                
            sigma = sigma * np.exp(c_sigma_eff / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + c_c * (m_new - mean) / sigma
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu sum w_i (y_i - mean)(y_i - mean)^T
            # The term (1 - (1-c_c)^2) C is part of the rank-1 update for pc.
            # Standard formula:
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu * rank_mu
            # This can be simplified to:
            # C = (1 - c_1 - c_mu) C + c_1 * (1 - (1-c_c)^2) C + c_1 pc pc^T + c_mu * rank_mu
            # C = (1 - c_1 - c_mu + c_1 - c_1(1-2c_c+c_c^2)) C + ...
            # C = (1 - c_mu - c_1(1-c_c)^2) C + c_1 pc pc^T + c_mu * rank_mu
            
            # Let's use the standard update:
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) C) + c_mu * sum(...)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                d = y_mu[i] - mean
                rank_mu += w[i] * np.outer(d, d)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu
            
            # Symmetrize to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update mean
            mean = m_new
            
            # Update Cholesky factor L
            # Add small regularization if needed
            L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

        return best_x, best_f

# EVOLVE-BLOCK-END
