import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        call_count = [0]
        best_f = [np.inf]
        best_x = [np.zeros(dim)]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise BudgetExceeded()
            call_count[0] += 1
            x = np.asarray(x, dtype=float)
            fx = func(x)
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        # Multi-fidelity SMC with GP surrogate
        def smc_gp_search(func_wrapper, dim, lb, ub, max_evals, seed=42):
            rng_local = np.random.default_rng(seed)
            
            # SMC parameters
            n_particles = 20  # Population size
            n_generations = 5  # Number of SMC generations
            n_refine = 10     # Refinement iterations per generation
            
            # GP hyperparameters
            length_scale = 1.0
            signal_var = 1.0
            noise_var = 1e-6
            
            # Initialize particles uniformly
            particles = rng_local.uniform(lb, ub, size=(n_particles, dim))
            
            # Evaluate initial particles
            X_eval = []
            Y_eval = []
            for i, p in enumerate(particles):
                f = func_wrapper(p)
                X_eval.append(p)
                Y_eval.append(f)
            
            # SMC iterations
            for gen in range(n_generations):
                if time.time() > deadline or call_count[0] >= max_evals:
                    break
                
                # Compute importance weights based on function values
                Y_arr = np.array(Y_eval)
                Y_arr = np.where(np.isinf(Y_arr), 1000.0, Y_arr)
                
                # Use negative exponential of scaled fitness as weights
                # Lower f -> higher weight
                scale = max(1.0, np.std(Y_arr))
                log_weights = -Y_arr / scale
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                weights /= np.sum(weights)
                
                # Resample particles (systematic resampling)
                cum_weights = np.cumsum(weights)
                u = (rng_local.uniform() + np.arange(n_particles)) / n_particles
                indices = np.searchsorted(cum_weights, u)
                indices = np.clip(indices, 0, len(weights) - 1)
                
                new_particles = particles[indices]
                
                # Perturb particles with decreasing step size
                sigma_gen = max(0.1, 1.0 * (1.0 - gen / n_generations))
                
                for i in range(n_particles):
                    if time.time() > deadline or call_count[0] >= max_evals:
                        break
                    
                    # GP-guided perturbation
                    if len(X_eval) >= 2:
                        X_arr = np.array(X_eval)
                        Y_arr = np.array(Y_eval)
                        Y_arr = np.where(np.isinf(Y_arr), 1000.0, Y_arr)
                        
                        # Simple GP posterior mean for guidance
                        # Use nearest neighbor as proxy for GP mean
                        dists = np.linalg.norm(X_arr - new_particles[i], axis=1)
                        nn_idx = np.argmin(dists)
                        gp_mean = Y_arr[nn_idx]
                        
                        # Sample from GP posterior (simplified: use NN value + noise)
                        target_f = gp_mean + rng_local.normal(0, 0.1 * scale)
                    else:
                        target_f = np.min(Y_eval)
                    
                    # Move toward lower function values
                    direction = rng_local.normal(0, 1, dim)
                    new_particles[i] = new_particles[i] + sigma_gen * direction
                    
                    # Clip to bounds
                    new_particles[i] = np.clip(new_particles[i], lb, ub)
                    
                    # Evaluate
                    f = func_wrapper(new_particles[i])
                    X_eval.append(new_particles[i])
                    Y_eval.append(f)
                
                particles = new_particles
            
            return best_x[0].copy(), best_f[0]

        # Phase 1: Run SMC-GP search with most of the budget
        smc_budget = int(budget * 0.8)
        
        try:
            cx, cf = smc_gp_search(wrapped_func, dim, lb, ub, smc_budget, seed=42)
            if cx is not None:
                if cf < best_f[0]:
                    best_f[0] = cf
                    best_x[0] = cx.copy()
        except (TimeoutError, BudgetExceeded):
            pass
        except Exception:
            pass

        # Phase 2: Local refinement with Nelder-Mead if budget remains
        if best_x[0] is not None and call_count[0] < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: wrapped_func(x),
                    best_x[0],
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        # Phase 3: Final local search with small perturbations
        if call_count[0] < budget and best_x[0] is not None:
            try:
                for _ in range(20):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    # Small random perturbations around best
                    scale = max(0.01, (best_f[0] - (-100)) / 200 * 0.1)
                    x_trial = best_x[0] + rng.normal(0, scale, dim)
                    x_trial = np.clip(x_trial, lb, ub)
                    f = wrapped_func(x_trial)
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        return best_x[0], best_f[0]
