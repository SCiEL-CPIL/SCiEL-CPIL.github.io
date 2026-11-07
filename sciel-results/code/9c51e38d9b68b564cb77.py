import numpy as np
import time
import random

class Optimizer:
    """
    Multi-Start Island Evolution Strategy for BBOB-style f17 (Schaffers F7).
    
    Strategy:
    - Use 3 independent 'islands' (populations).
    - Each island uses a simplified CMA-ES-like adaptation (or just adaptive sigma) 
      but with a very small population to save budget.
    - Migration: Every K generations, the best solution from one island is copied to another.
    - This helps escape local minima (Schaffers F7 has many) and handles rotation 
      by allowing different islands to explore different orientations initially.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        best_x_global = None
        best_f_global = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x_global, best_f_global
            if calls_made >= total_budget:
                return float('inf')
            try:
                f = func(x)
                calls_made += 1
                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x.copy()
                return f
            except Exception:
                return float('inf')

        # Initialize Islands
        num_islands = 3
        # Small population per island to allow more generations
        # Total budget 500. 3 islands. 
        # If pop_size=4, generations per island ~ 500/12 = 41.
        # Let's use pop_size=5 for slightly better averaging, 500/15=33 gens.
        # Or pop_size=4 is faster. Let's try 4.
        pop_size = 4
        mu = 2 # Select top 2
        
        islands = []
        for i in range(num_islands):
            # Initialize mean at different random points to explore landscape
            # Box is [-5, 5]^5
            init_mean = np.random.uniform(-3, 3, dim)
            init_sigma = 1.5 # Start wide
            
            # Simple covariance: diagonal only for speed? 
            # Schaffers F7 is rotation-invariant in structure but rotated. 
            # Full C is better for rotation. Let's use a lightweight CMA-ES.
            C = np.eye(dim)
            
            island = {
                'mean': init_mean,
                'sigma': init_sigma,
                'C': C,
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'best_x': init_mean.copy(),
                'best_f': float('inf'),
                'pop': [],
                'fit': []
            }
            islands.append(island)
            
            # Evaluate initial mean for each island to set best
            f = safe_eval(island['mean'])
            island['best_f'] = f
            island['best_x'] = island['mean'].copy()

        # CMA-ES parameters for dim=5
        # lambda = 4, mu = 2
        weights = np.array([1.0, 0.5])
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        migration_interval = 5 # Generations
        
        gen = 0
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
                
            # For each island, generate next population
            for i, island in enumerate(islands):
                if calls_made >= total_budget:
                    break
                
                # Sample population
                pop = []
                fits = []
                
                for k in range(pop_size):
                    # Sample from N(mean, sigma^2 * C)
                    try:
                        L = np.linalg.cholesky(island['C'])
                    except np.linalg.LinAlgError:
                        island['C'] = np.eye(dim)
                        L = np.eye(dim)
                        
                    z = np.random.randn(dim)
                    x = island['mean'] + island['sigma'] * (L @ z)
                    x = np.clip(x, -5, 5)
                    
                    f = safe_eval(x)
                    pop.append(x)
                    fits.append(f)
                    
                    if calls_made >= total_budget:
                        break
                
                if len(pop) < mu:
                    continue
                    
                # Sort
                idx = np.argsort(fits)
                pop = [pop[j] for j in idx]
                fits = [fits[j] for j in idx]
                
                # Update best for island
                if fits[0] < island['best_f']:
                    island['best_f'] = fits[0]
                    island['best_x'] = pop[0].copy()
                
                # Update CMA-ES state
                # Weighted mean
                new_mean = np.zeros(dim)
                for j in range(mu):
                    new_mean += weights[j] * pop[j]
                
                old_mean = island['mean']
                island['mean'] = new_mean
                
                # Update sigma and C
                y = (new_mean - old_mean) / (island['sigma'] + 1e-12)
                
                # p_sigma update
                island['p_sigma'] = (1 - c_sigma) * island['p_sigma'] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                h_sigma = (np.linalg.norm(island['p_sigma']) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / \
                          (np.sqrt(mu_eff) * np.sqrt(np.trace(island['C'])))
                
                # p_c update
                island['p_c'] = (1 - c_c) * island['p_c'] + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                h_c = (np.linalg.norm(island['p_c']) / np.sqrt(1 - (1 - c_c) ** (gen + 1))) / \
                      (np.sqrt(mu_eff) * np.sqrt(np.trace(island['C'])))
                
                # Sigma update
                island['sigma'] = island['sigma'] * np.exp(c_sigma / d_sigma * (np.linalg.norm(island['p_sigma']) / np.sqrt(mu_eff) - 1))
                
                # C update
                rank_one = np.outer(y, y)
                rank_mu = np.zeros((dim, dim))
                for j in range(mu):
                    y_j = (pop[j] - old_mean) / (island['sigma'] + 1e-12)
                    rank_mu += weights[j] * np.outer(y_j, y_j)
                
                delta_h = (1 - h_sigma) * c_sigma * (2 - c_sigma)
                delta_hc = h_c * c_c * (2 - c_c) - 1
                
                island['C'] = (1 + c_1 * delta_hc + c_mu * (delta_h - delta_hc)) * island['C'] \
                              - c_1 * rank_one + c_mu * rank_mu
                              
                # Symmetrize and ensure PSD
                island['C'] = (island['C'] + island['C'].T) / 2
                # Regularize if needed
                # Simple check: eigenvalues
                try:
                    eigvals = np.linalg.eigvalsh(island['C'])
                    if np.any(eigvals < 1e-8):
                         island['C'] += 1e-6 * np.eye(dim)
                except:
                    pass

            # Migration
            gen += 1
            if gen % migration_interval == 0:
                # Find best island
                best_island_idx = np.argmin([isl['best_f'] for isl in islands])
                # Migrate to a random other island
                if len(islands) > 1:
                    others = [j for j in range(num_islands) if j != best_island_idx]
                    target_idx = random.choice(others)
                    # Replace worst individual in target or just update mean?
                    # Simple: Replace the mean of the target with the best x of the source
                    # This is a strong migration.
                    islands[target_idx]['mean'] = islands[best_island_idx]['best_x'].copy()
                    # Optionally reset sigma to explore around the new mean
                    # islands[target_idx]['sigma'] = 1.0 
                    # Or keep sigma to continue exploiting. Let's keep sigma.
                    
        return best_x_global, best_f_global
