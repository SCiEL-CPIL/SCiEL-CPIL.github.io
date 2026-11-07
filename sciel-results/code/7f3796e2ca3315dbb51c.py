"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_sequence(n, dim, seed=42):
    """Generate a Sobol sequence using a simple bit-reversal approximation or fallback to LHS if scipy unavailable/slow.
    For this task, we use a standard Sobol generator if available, otherwise a structured LHS.
    Since scipy is available, we use scipy.stats.qmc.Sobol.
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    # We need to generate enough points. Sobol requires power of 2 for best efficiency, 
    # but we can generate a bit more and slice.
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    points = sampler.random(next_pow2)
    return points[:n]


class Optimizer:
    """Two-stage optimizer: Sobol exploration followed by CMA-ES exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Stage 1: Global Exploration with Sobol Sequence
        # We use the first 100 calls for a structured global search.
        # This helps escape local minima of Rastrigin more effectively than random init.
        n_explore = 100
        if budget < n_explore:
            n_explore = budget
            
        points = _sobol_sequence(n_explore, dim, seed=123)
        # Map from [0,1]^dim to [-5,5]^dim
        points = points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if something went wrong or no calls were made
            x0 = rng.uniform(-5, 5, dim)
            best_f = func(x0)
            best_x = x0
            calls = 1

        # Stage 2: Local Exploitation with CMA-ES
        # We implement a simplified CMA-ES to avoid heavy dependencies if scipy.optimize.cma is slow or unavailable in this context,
        # but since scipy is available, we can use scipy.optimize.minimize with method='Nelder-Mead' as a robust local search,
        # OR implement a lightweight CMA-ES.
        # Given the "rotated" nature, CMA-ES is theoretically better. Let's implement a compact CMA-ES.
        
        # CMA-ES Parameters
        pop_size = max(4, int(4 + 3 * np.log(dim))) # Standard CMA-ES population size
        sigma = 1.0  # Initial step size
        # We will use the remaining budget
        remaining_budget = budget - calls
        
        # CMA-ES State
        mean = best_x.copy()
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim) # Eigenvalues (std devs)
        
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1/mu_eff)) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # We need to be careful with the number of generations.
        # Each generation costs pop_size evaluations.
        # We stop when budget is exhausted or time is up.
        
        while calls + pop_size <= budget:
            if time.time() > deadline:
                break
                
            # Generate population
            pop = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # Transform z using C
                # y = B * (D * z)
                y = B @ (D * z)
                x = mean + sigma * y
                pop.append(x)
                
            # Evaluate
            fvals = []
            for x in pop:
                if time.time() > deadline:
                    break
                fx = func(x)
                calls += 1
                fvals.append(fx)
                
            if len(fvals) < pop_size:
                # If we ran out of time mid-generation, just keep the best so far
                # and break out to return
                break
                
            # Sort by fitness
            indices = np.argsort(fvals)
            pop_sorted = [pop[i] for i in indices]
            fvals_sorted = [fvals[i] for i in indices]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Update mean
            new_mean = np.sum([weights[i] * pop_sorted[i] for i in range(mu)], axis=0)
            
            # Update step size control vector p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * C_inv^0.5 * (new_mean - mean) / sigma
            # C_inv^0.5 is B @ diag(1/D) @ B.T
            C_inv_sqrt = B @ (B.T @ (1.0/D)) # Wait, B is orthogonal, so B.T @ B = I. 
            # Actually, C = B @ diag(D^2) @ B.T.
            # C_inv^0.5 = B @ diag(1/D) @ B.T
            C_inv_sqrt = B @ (np.diag(1.0/D) @ B.T)
            
            diff = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ diff)
            
            # Hessian update for covariance
            # If ||p_sigma|| < (1.4 + 1/dim) * chi_n, update p_c
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 1.0/dim) * chi_n:
                p_c = (1 - c_c) * p_c + c_c * (new_mean - mean) / sigma
                
            # Rank-one and rank-mu updates
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i * y_i.T)
            h_c = (1 - (1 - c_c)**2) / (dim * (2 - c_c)) if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 1.0/dim) * chi_n else 1.0
            
            # Compute y_i for top mu
            y_vecs = []
            for i in range(mu):
                y_i = (pop_sorted[i] - mean) / sigma
                y_vecs.append(y_i)
                
            rank_one = p_c * p_c.T + (1 - h_c) * c_c * (2 - c_c) * C
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y_vecs[i], y_vecs[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update eigenvalues/vectors periodically to maintain numerical stability
            # We re-diagonalize every 10 generations or so
            # For simplicity and speed in this constrained environment, we might just update D and B every generation if dim is small (5)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Ensure positive
            D = np.sqrt(eigvals)
            B = eigvecs
            
            mean = new_mean
            
            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            if sigma < 1e-12:
                break
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f

# EVOLVE-BLOCK-END
