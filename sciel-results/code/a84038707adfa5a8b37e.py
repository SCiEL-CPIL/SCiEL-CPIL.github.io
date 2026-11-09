"""CMA-ES with multi-start for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for step ellipsoidal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        # Phase 1: Global Scout to find the 'step' region
        # f7 has a sharp drop. We need to find a point inside the smooth ellipsoidal region.
        # We use a small budget to scan broadly.
        n_scout = min(20, budget // 5)
        
        # Candidate starts for CMA-ES
        starts = []
        
        for i in range(n_scout):
            if calls_made >= budget or time_left() < 50.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Heuristic: If the function value is significantly lower than the initial average 
            # or just generally low, it might be in the 'good' region.
            # For f7, values drop from ~100+ to ~0-10. 
            # We'll collect the top few distinct regions.
            if len(starts) < 5:
                # Check if this point is far enough from existing starts to be a new basin
                is_new = True
                for s in starts:
                    if np.linalg.norm(x - s[0]) < 1.0:
                        is_new = False
                        break
                if is_new:
                    starts.append((x.copy(), fx))
            else:
                # Replace the worst start if this one is better
                worst_idx = max(range(len(starts)), key=lambda i: starts[i][1])
                if fx < starts[worst_idx][1]:
                    starts[worst_idx] = (x.copy(), fx)

        # Phase 2: CMA-ES Refinement
        # Allocate remaining budget to CMA-ES runs
        # We will run CMA-ES on the best start found, and potentially others if budget allows.
        
        # Simple CMA-ES implementation
        def run_cmaes(x0, f0, max_calls):
            nonlocal calls_made, best_f, best_x
            
            if max_calls <= 0:
                return x0, f0

            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))
            sigma = 1.0
            mean = x0.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            # Adaptation parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff / 2)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            x_best_local = x0.copy()
            f_best_local = f0
            
            for i_iter in range(max_calls // pop_size):
                if calls_made >= budget or time_left() < 5.0:
                    break
                
                # Generate population
                for i in range(pop_size):
                    if calls_made >= budget or time_left() < 5.0:
                        break
                    
                    # Sample from multivariate normal
                    z = rng.standard_normal(dim)
                    # Transform using C^(1/2)
                    # For simplicity and speed in low dim, we can just use diagonal approximation or 
                    # standard normal scaled by sigma. 
                    # Proper CMA-ES requires Cholesky.
                    L = np.linalg.cholesky(C)
                    y = L @ z
                    x_i = mean + sigma * y
                    x_i = np.clip(x_i, -5, 5)
                    
                    f_i = func(x_i)
                    calls_made += 1
                    
                    if f_i < f_best_local:
                        f_best_local = f_i
                        x_best_local = x_i.copy()
                        if f_i < best_f:
                            best_f = f_i
                            best_x = x_i.copy()
                
                # Sort by fitness
                # We need to store the evaluated points
                # Note: The loop above doesn't store all points efficiently for sorting.
                # Let's refactor CMA-ES to store points in a list per generation.
                
                # REFACTORING CMA-ES INTERNALS FOR CORRECTNESS
                # The above loop is slightly flawed for strict CMA-ES as it doesn't keep 
                # the population for sorting/adaptation. Let's implement a cleaner version.
                
                pass # Placeholder, will replace with correct implementation below

        # Correct CMA-ES Implementation
        def cmaes_search(x0, f0, max_iter):
            nonlocal calls_made, best_f, best_x
            
            if max_iter <= 0:
                return x0, f0

            pop_size = 4 + int(3 * np.log(dim))
            sigma = 1.0
            mean = x0.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff / 2)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            x_best_local = x0.copy()
            f_best_local = f0
            
            for i_iter in range(max_iter):
                if calls_made >= budget or time_left() < 5.0:
                    break
                
                # Generate population
                pop_x = []
                pop_f = []
                
                for i in range(pop_size):
                    if calls_made >= budget or time_left() < 5.0:
                        break
                    
                    z = rng.standard_normal(dim)
                    L = np.linalg.cholesky(C)
                    y = L @ z
                    x_i = mean + sigma * y
                    x_i = np.clip(x_i, -5, 5)
                    
                    f_i = func(x_i)
                    calls_made += 1
                    
                    pop_x.append(x_i)
                    pop_f.append(f_i)
                    
                    if f_i < f_best_local:
                        f_best_local = f_i
                        x_best_local = x_i.copy()
                        if f_i < best_f:
                            best_f = f_i
                            best_x = x_i.copy()
                
                if len(pop_x) < mu:
                    break
                
                # Sort
                idx = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in idx]
                pop_f = [pop_f[i] for i in idx]
                
                # Weighted recombination
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * pop_x[i]
                
                # Update mean
                mean = mean + sigma * y_w
                
                # Update step size
                y_old = np.linalg.norm(y_w)
                if y_old > 1e-9:
                    cs = (1 - c_sigma) * cs + c_sigma * (y_old / chi_n - 1)
                    sigma *= np.exp(c_sigma / d_sigma * (y_old / chi_n - 1))
                
                # Update covariance
                delta_s = (1 if y_old / chi_n > 1.4 + 2 / (dim + 1) else 0)
                pc = (1 - c_c) * pc + c_c * (delta_s * y_w / sigma)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - delta_s * c_c * (2 - c_c)) * np.eye(dim))
                
                # Rank-mu update
                for i in range(mu):
                    y_i = (pop_x[i] - mean + sigma * y_w) / sigma # Approximation, should be (x_i - mean)/sigma
                    # Correct y_i calculation:
                    y_i = (pop_x[i] - (mean - sigma * y_w)) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-6 * np.eye(dim)
                
                # Reset sigma if it becomes too small or large
                if sigma < 1e-10:
                    sigma = 1e-10
                if sigma > 100:
                    sigma = 100

            return x_best_local, f_best_local

        # Run CMA-ES on the best start
        if best_x is not None:
            remaining_budget = budget - calls_made
            # Allocate most of the budget to the best start
            max_iter = remaining_budget // (4 + int(3 * np.log(dim)))
            if max_iter > 0 and time_left() > 10.0:
                cmaes_search(best_x, best_f, max_iter)
            
            # If we have budget left and found multiple good starts, try the second best
            if len(starts) > 1 and calls_made < budget - 50 and time_left() > 20.0:
                second_best = starts[1][0]
                # Ensure it's not the same as best
                if np.linalg.norm(second_best - best_x) > 0.5:
                    remaining_budget = budget - calls_made
                    max_iter = remaining_budget // (4 + int(3 * np.log(dim)))
                    if max_iter > 0:
                        cmaes_search(second_best, starts[1][1], max_iter)

        return best_x, best_f
# EVOLVE-BLOCK-END
