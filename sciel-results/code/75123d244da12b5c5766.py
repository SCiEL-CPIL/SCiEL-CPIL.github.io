"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Island-based CMA-ES with migration for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
        
        # Number of islands
        n_islands = 4
        lambda_ = max(4, int(np.ceil(1 + 3 * np.log(dim))))  # population size per island
        
        # Initialize islands with diverse starting points
        islands = []
        for i in range(n_islands):
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            calls_made += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Initialize CMA-ES state for this island
            sigma = 2.0  # initial step size
            mean = x0.copy()
            C = np.eye(dim)
            Cs = np.zeros((dim, dim))
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Compute inverse square root of C for weight normalization
            # For simplicity, use Cholesky decomposition
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            C_inv_sqrt = np.linalg.inv(L)
            
            # Compute weights
            mu = lambda_ // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            islands.append({
                'mean': mean,
                'sigma': sigma,
                'C': C,
                'Cs': Cs,
                'ps': ps,
                'pc': pc,
                'C_inv_sqrt': C_inv_sqrt,
                'weights': weights,
                'mu_eff': mu_eff,
                'c_sigma': c_sigma,
                'c_c': c_c,
                'c_1': c_1,
                'c_mu': c_mu,
                'chi_n': chi_n,
                'local_best_x': x0.copy(),
                'local_best_f': f0,
                'calls_used': 1,
                'stagnation': 0,
                'migration_counter': 0
            })
        
        def cma_es_step(island, func, max_calls, deadline, dim, rng):
            """
            Perform one generation of CMA-ES for a single island.
            Returns number of function calls made.
            """
            if island['calls_used'] >= max_calls or time_left() < 0.5:
                return 0
            
            mean = island['mean']
            sigma = island['sigma']
            C = island['C']
            Cs = island['Cs']
            ps = island['ps']
            pc = island['pc']
            C_inv_sqrt = island['C_inv_sqrt']
            weights = island['weights']
            mu_eff = island['mu_eff']
            c_sigma = island['c_sigma']
            c_c = island['c_c']
            c_1 = island['c_1']
            c_mu = island['c_mu']
            chi_n = island['chi_n']
            lambda_ = len(island['weights']) * 2  # approximate, should be stored
            
            # Actually, let's store lambda_ in the island
            if 'lambda_' not in island:
                island['lambda_'] = lambda_
            lambda_ = island['lambda_']
            
            # Sample lambda_ individuals
            z = rng.standard_normal((lambda_, dim))
            y = z @ C_inv_sqrt
            x = mean + sigma * y.T
            x = np.clip(x, -5, 5)
            
            # Evaluate
            f = np.array([func(xi) for xi in x])
            calls_made_local = lambda_
            
            # Update global best
            if f.min() < best_f:
                best_f = f.min()
                best_x = x[f.argmin()].copy()
            
            # Sort by fitness
            order = np.argsort(f)
            x = x[order]
            f = f[order]
            
            # Update local best
            if f[0] < island['local_best_f']:
                island['local_best_f'] = f[0]
                island['local_best_x'] = x[0].copy()
                island['stagnation'] = 0
            else:
                island['stagnation'] += 1
            
            # Weighted mean
            mu = len(weights)
            w = weights[:mu]
            y_w = np.sum(w[:, None] * y[order[:mu]], axis=0)
            x_w = np.sum(w[:, None] * x[order[:mu]], axis=0)
            
            # Update covariance matrix
            C = (1 + c_1 * c_mu * (mu_eff - 2) / dim - c_1 - c_mu) * C + \
                c_1 * (np.outer(pc, pc) - Cs) + \
                c_mu * np.sum(w[:, None, None] * np.outer(y[order[:mu]], y[order[:mu]]), axis=0)
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update Cs
            Cs = (1 - c_c) * Cs + c_c * (2 - c_c) * (np.outer(pc, pc) - Cs)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * y_w
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w
            
            # Update sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = x_w
            
            # Update C_inv_sqrt
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            C_inv_sqrt = np.linalg.inv(L)
            
            # Update island state
            island['mean'] = mean
            island['sigma'] = sigma
            island['C'] = C
            island['Cs'] = Cs
            island['ps'] = ps
            island['pc'] = pc
            island['C_inv_sqrt'] = C_inv_sqrt
            island['calls_used'] += calls_made_local
            
            return calls_made_local
        
        # Main loop: run CMA-ES generations across islands
        generation = 0
        while calls_made < budget and time_left() > 1.0:
            # For each island, run one CMA-ES generation
            for i in range(n_islands):
                if calls_made >= budget or time_left() < 0.5:
                    break
                
                used = cma_es_step(islands[i], func, budget, deadline, dim, rng)
                calls_made += used
                
                # Migration: if an island stagnates, migrate from the best island
                if islands[i]['stagnation'] > 10 and i != 0:
                    # Find the best island
                    best_island_idx = np.argmin([isl['local_best_f'] for isl in islands])
                    if best_island_idx != i:
                        # Migrate: replace this island's mean with the best island's mean
                        islands[i]['mean'] = islands[best_island_idx]['mean'].copy()
                        islands[i]['sigma'] = islands[best_island_idx]['sigma']
                        islands[i]['stagnation'] = 0
                        islands[i]['migration_counter'] += 1
            
            generation += 1
            
            # Every 20 generations, do a random restart for the worst island if it's stuck
            if generation % 20 == 0:
                worst_idx = np.argmax([isl['stagnation'] for isl in islands])
                if islands[worst_idx]['stagnation'] > 30:
                    # Random restart
                    x0 = rng.uniform(-5, 5, dim)
                    f0 = func(x0)
                    calls_made += 1
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    
                    islands[worst_idx]['mean'] = x0.copy()
                    islands[worst_idx]['sigma'] = 2.0
                    islands[worst_idx]['C'] = np.eye(dim)
                    islands[worst_idx]['Cs'] = np.zeros((dim, dim))
                    islands[worst_idx]['ps'] = np.zeros(dim)
                    islands[worst_idx]['pc'] = np.zeros(dim)
                    L = np.linalg.cholesky(np.eye(dim) + 1e-10 * np.eye(dim))
                    islands[worst_idx]['C_inv_sqrt'] = np.linalg.inv(L)
                    islands[worst_idx]['local_best_f'] = f0
                    islands[worst_idx]['local_best_x'] = x0.copy()
                    islands[worst_idx]['stagnation'] = 0
        
        return best_x, best_f

# EVOLVE-BLOCK-END
