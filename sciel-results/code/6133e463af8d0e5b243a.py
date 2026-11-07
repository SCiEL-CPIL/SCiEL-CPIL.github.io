"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_iter, rng):
            nonlocal calls
            if calls >= budget or not time_ok():
                return None, np.inf
            
            dim = len(x0)
            x = x0.copy()
            sigma = sigma0
            best_x_local = x.copy()
            best_f_local = np.inf
            
            # CMA-ES parameters
            mu = max(1, dim // 2)  # Number of parents
            lambda_ = 2 * mu  # Population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step size adaptation
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Mean adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Covariance matrix
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Initial evaluation
            f_x = func(x)
            calls += 1
            record(x, f_x)
            best_f_local = f_x
            best_x_local = x.copy()
            
            for iteration in range(max_iter):
                if not time_ok() or calls >= budget:
                    break
                
                # Generate population
                pop = []
                pop_f = []
                
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    pop.append(y)
                    
                    if calls >= budget or not time_ok():
                        break
                    f_y = func(y)
                    calls += 1
                    record(y, f_y)
                    pop_f.append(f_y)
                
                if len(pop) < 2:
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(pop_f)
                pop = [pop[i] for i in sorted_idx]
                pop_f = [pop_f[i] for i in sorted_idx]
                
                # Update best
                if pop_f[0] < best_f_local:
                    best_f_local = pop_f[0]
                    best_x_local = pop[0].copy()
                
                # Select top mu individuals
                selected = pop[:mu]
                selected_f = pop_f[:mu]
                
                # Weighted mean
                x_new = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)
                
                # Update covariance matrix
                y_selected = np.array([(selected[i] - x) / sigma for i in range(mu)])
                delta_x = x_new - x
                
                # Update evolution path for sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (delta_x / sigma))
                
                # h_sigma condition
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                
                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta_x / sigma)
                
                # Rank-one and rank-mu updates
                C = (1 + c_1 * h_sigma * (h_sigma - 1 + c_c)) * C \
                    + c_1 * (h_sigma * (p_c @ p_c - C)) \
                    + c_mu * np.sum([weights[i] * np.outer(y_selected[i], y_selected[i]) for i in range(mu)], axis=0)
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update inverse and square root
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                
                # Update mean
                x = x_new
                
                # Check convergence
                if np.max(pop_f[:mu]) - np.min(pop_f[:mu]) < 1e-10:
                    break
                if sigma < 1e-10:
                    break
            
            return best_x_local, best_f_local

        # Phase 1: Multiple CMA-ES restarts from different starting points
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        num_restarts = min(5, budget // 50)
        starts = []
        
        # Include origin and some random points
        starts.append(np.zeros(dim))
        for _ in range(num_restarts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Also add points based on grid
        for i in range(dim):
            for val in [-3, 3]:
                p = np.zeros(dim)
                p[i] = val
                starts.append(p)
        
        # Shuffle starts
        rng.shuffle(starts)
        
        # Run CMA-ES from each start
        for start in starts:
            if not time_ok() or calls >= budget:
                break
            
            # Adaptive sigma based on budget remaining
            remaining = budget - calls
            max_iter = min(50, remaining // 5)
            sigma0 = 1.5
            
            x_best_local, f_best_local = cma_es(start, sigma0, max_iter, rng)
            
            if x_best_local is not None and f_best_local < best_f:
                best_f = f_best_local
                best_x = x_best_local.copy()
            
            if not time_ok() or calls >= budget:
                break
        
        # Phase 2: Final local refinement with Nelder-Mead if budget remains
        if time_ok() and calls < budget and best_x is not None:
            x0 = best_x.copy()
            
            # Try a few different initial simplex sizes
            for alpha in [0.2, 0.5, 1.0]:
                if not time_ok() or calls >= budget:
                    break
                
                simplex = [x0.copy()]
                f_simplex = [best_f]
                
                for i in range(dim):
                    if not time_ok() or calls >= budget:
                        break
                    x_i = x0.copy()
                    x_i[i] += alpha
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls += 1
                    simplex.append(x_i)
                    f_simplex.append(f_i)
                    record(x_i, f_i)
                    if not time_ok():
                        break
                
                if len(simplex) < 2:
                    continue
                
                # Nelder-Mead parameters
                gamma = 1.0
                rho = 2.0
                sigma_nm = 0.5
                shrink = 0.5
                
                nm_iterations = 0
                max_nm_iter = min(30, (budget - calls) // 2)
                
                while nm_iterations < max_nm_iter and calls < budget and time_ok():
                    nm_iterations += 1
                    # Sort simplex by function value
                    indices = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in indices]
                    f_simplex = [f_simplex[i] for i in indices]
                    
                    x_best = simplex[0]
                    f_best = f_simplex[0]
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    x_second_worst = simplex[-2]
                    f_second_worst = f_simplex[-2]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    if not time_ok() or calls >= budget:
                        break
                    f_r = func(x_r)
                    calls += 1
                    record(x_r, f_r)
                    
                    if f_r < f_best:
                        # Expansion
                        x_e = centroid + rho * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if not time_ok() or calls >= budget:
                            break
                        f_e = func(x_e)
                        calls += 1
                        record(x_e, f_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_second_worst:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        # Contraction
                        if f_r < f_worst:
                            # Outside contraction
                            x_c = centroid + sigma_nm * (x_r - centroid)
                        else:
                            # Inside contraction
                            x_c = centroid - sigma_nm * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        
                        if not time_ok() or calls >= budget:
                            break
                        f_c = func(x_c)
                        calls += 1
                        record(x_c, f_c)
                        
                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if not time_ok() or calls >= budget:
                                    break
                                simplex[i] = x_best + shrink * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                                record(simplex[i], f_simplex[i])
                            if not time_ok():
                                break
                    
                    # Check for convergence
                    if np.max(f_simplex) - np.min(f_simplex) < 1e-12:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
