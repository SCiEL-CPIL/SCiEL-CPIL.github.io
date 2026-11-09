# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid-scan followed by CMA-ES for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse Grid Scan
        # Use 3 points per dimension: -4, 0, 4
        # Total points: 3^5 = 243
        grid_points = [-4.0, 0.0, 4.0]
        
        # Generate grid
        grid = np.array(np.meshgrid(*([grid_points] * dim))).T.reshape(-1, dim)
        
        # Shuffle grid to ensure early progress score (AOCC) benefits from random-ish order?
        # No, deterministic is requested. But shuffling helps if the best point is late in lexicographic order.
        # Let's shuffle with a fixed seed for reproducibility but "random" order for AOCC.
        perm = rng.permutation(len(grid))
        grid = grid[perm]

        for point in grid:
            if calls >= budget or time.time() > deadline:
                break
            fx = evaluate(point)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = point.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES
        # Initialize CMA-ES around the best grid point
        x = best_x.copy()
        sigma = 1.0 # Initial step size
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim)) # Population size
        mu = lambda_ // 2 # Weighting threshold
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # Track best
        best_x_cma = best_x.copy()
        best_f_cma = best_f
        
        # Main CMA-ES Loop
        while calls < budget and time.time() < deadline:
            # Generate population
            offspring = []
            fitnesses = []
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                xi = x + sigma * z
                xi = np.clip(xi, -5.0, 5.0)
                fx = evaluate(xi)
                if fx is None:
                    break
                offspring.append(xi)
                fitnesses.append(fx)
                
                if fx < best_f_cma:
                    best_f_cma = fx
                    best_x_cma = xi.copy()
                if fx < best_f:
                    best_f = fx
                    best_x = xi.copy()
            
            if len(offspring) < lambda_:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fitnesses = [fitnesses[i] for i in indices]
            
            # Update mean
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            
            # Update sigma (CSA)
            y_c = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls - self.last_sigma_update if hasattr(self, 'last_sigma_update') else 0))) < (1.4 + 2 / (dim + 1)) * chi_N
            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update Covariance (CSA)
            if h_sigma:
                delta_h = (1 - h_sigma) * (1 - c_c) - 1
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            else:
                rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + c_mu * rank_mu
            
            # Update square roots
            # Simple eigenvalue decomposition for dim=5 is fast
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
            
            x = x_new
            
            # Termination condition: sigma too small
            if sigma < 1e-6:
                break
                
            self.last_sigma_update = calls

        return best_x, best_f
# EVOLVE-BLOCK-END
