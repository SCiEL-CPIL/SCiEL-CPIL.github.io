"""CMA-ES optimizer with adaptive distribution for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to estimate center and scale
        n_init = 10
        X_init = []
        Y_init = []
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f_val, x_eval = call_func(x)
            if f_val is not None:
                X_init.append(x_eval)
                Y_init.append(f_val)
        
        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None, np.inf

        # Initialize CMA-ES parameters
        # Mean is the best point found so far
        mean = best_x.copy()
        
        # Initial sigma: estimate from spread of initial points or use a heuristic
        # Weierstrass has features at multiple scales. Start with a moderate sigma.
        sigma = 1.0 
        
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        if lambda_pop < 8:
            lambda_pop = 8
        if lambda_pop > 20:
            lambda_pop = 20
            
        # CMA-ES parameters
        c_sigma = (lambda_pop + 2) / (dim + lambda_pop + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((lambda_pop / 2 + 1) / (dim + 2) - 1))
        
        # Cumulation path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Eigen-decomposition cache
        C = np.eye(dim)
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def update_CMA(mean, sigma, C, D, B, pc, ps, x_samples, f_values):
            nonlocal c_sigma, d_sigma, c_c, c_1, c_mu, chi_n, lambda_pop, dim
            
            # Sort by fitness
            order = np.argsort(f_values)
            x_sorted = x_samples[order]
            f_sorted = f_values[order]
            
            # Weights for mean update
            mu_eff = np.sum(np.exp(-2 * np.arange(lambda_pop // 2) / (lambda_pop // 2)))
            weights = np.zeros(lambda_pop)
            weights[:lambda_pop // 2] = np.exp(-2 * np.arange(lambda_pop // 2) / (lambda_pop // 2))
            weights /= np.sum(weights)
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * x_sorted, axis=0)
            
            # Update sigma (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / D @ (B.T @ (x_sorted[0] - mean) / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (np.linalg.norm(ps) > (1.4 + 0.5 / c_c) * chi_n) * ps / sigma
            
            # Rank-mu update
            v = (x_sorted[:lambda_pop // 2] - mean) / sigma
            rank_mu = np.sum(weights[:lambda_pop // 2][:, np.newaxis, np.newaxis] * v[:, :, np.newaxis] * v[:, np.newaxis, :], axis=0)
            
            # Rank-one update
            y = (new_mean - mean) / sigma
            rank_one = np.outer(y, y)
            
            C = (1 + c_1 * (np.trace(pc) / dim - 1) + c_mu - c_1 - c_mu) * C + c_1 * pc + c_mu * rank_mu
            
            # Ensure positive definite
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-10))
                B = B2
                D = D2
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                
            return new_mean, sigma, C, D, B, pc, ps

        # Main CMA-ES Loop
        # We need to be careful with the number of evaluations per generation.
        # Budget is 500. n_init=10. Remaining ~490.
        # lambda_pop ~ 10-12.
        
        while n_calls < budget and not check_time():
            # Generate population
            x_samples = []
            f_values = []
            
            for i in range(lambda_pop):
                if check_time() or n_calls >= budget:
                    break
                z = rng.standard_normal(dim)
                x_sample = mean + sigma * (B @ (D * z))
                x_sample = np.clip(x_sample, -5, 5)
                
                f_val, x_eval = call_func(x_sample)
                if f_val is not None:
                    x_samples.append(x_eval)
                    f_values.append(f_val)
            
            if len(x_samples) == 0:
                break
                
            x_samples = np.array(x_samples)
            f_values = np.array(f_values)
            
            # Update CMA-ES state
            mean, sigma, C, D, B, pc, ps = update_CMA(mean, sigma, C, D, B, pc, ps, x_samples, f_values)
            
            # If sigma becomes very small, we might have converged
            if sigma < 1e-4:
                break

        # Phase 3: Local Polish with Nelder-Mead or small CMA-ES
        # If we have budget left, do a fine-grained search around best_x
        if n_calls < budget and not check_time():
            # Use a simple local search
            # Nelder-Mead is good here
            x_best = best_x.copy()
            f_best = best_f
            
            # Create simplex
            delta = 0.1
            simplex = [x_best.copy()]
            for i in range(dim):
                v = x_best.copy()
                v[i] += delta
                simplex.append(v)
            
            simplex = np.array(simplex)
            f_simplex = np.array([best_f])
            for i in range(1, dim + 1):
                f_val, _ = call_func(simplex[i])
                if f_val is not None:
                    f_simplex = np.append(f_simplex, f_val)
                else:
                    f_simplex = np.append(f_simplex, np.inf)
            
            # Nelder-Mead iterations
            max_iter = 50
            for _ in range(max_iter):
                if check_time() or n_calls >= budget:
                    break
                # Sort
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-8:
                    break
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = x_centroid + (x_centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect, _ = call_func(x_reflect)
                if f_reflect is None: f_reflect = np.inf
                
                if f_reflect < f_simplex[0]:
                    # Expansion
                    x_expand = x_centroid + 2 * (x_reflect - x_centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand, _ = call_func(x_expand)
                    if f_expand is None: f_expand = np.inf
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[-2]:
                    simplex[-1] = x_reflect
                    f_simplex[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = x_centroid + 0.5 * (simplex[-1] - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract, _ = call_func(x_contract)
                    if f_contract is None: f_contract = np.inf
                    
                    if f_contract < f_simplex[-1]:
                        simplex[-1] = x_contract
                        f_simplex[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_val, _ = call_func(simplex[i])
                            if f_val is not None:
                                f_simplex[i] = f_val
                            else:
                                f_simplex[i] = np.inf
        
        return best_x, best_f
# EVOLVE-BLOCK-END
