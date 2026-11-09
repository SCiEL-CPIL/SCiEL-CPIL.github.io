import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start Adaptive Simulated Annealing (ASA)
        # ASA uses a deterministic update direction based on the previous step,
        # combined with temperature-controlled random perturbations to escape local minima.
        # This is a structural change from CMA-ES, using a single trajectory with memory
        # of the search direction rather than a population-based covariance matrix.

        # Number of restarts for multi-start
        n_restarts = 8
        evals_per_restart = max(1, budget // n_restarts)
        
        for restart in range(n_restarts):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Initialize position
            if restart == 0:
                x = np.zeros(dim)
            else:
                # Start near the best found so far with some perturbation
                if best_x is not None:
                    x = best_x + rng.standard_normal(dim) * 0.5
                else:
                    x = rng.uniform(-4, 4, size=dim)
            
            fx = try_call(x)
            if fx is None:
                break
            
            # ASA parameters
            # T0: initial temperature, T_end: final temperature
            # The temperature controls the magnitude of random perturbations
            T0 = 1.0
            T_end = 1e-4
            # Cooling rate: T_{k+1} = T_k * alpha
            # We want T to go from T0 to T_end over evals_per_restart steps
            n_steps = evals_per_restart
            if n_steps > 1:
                alpha_cool = (T_end / T0) ** (1.0 / (n_steps - 1))
            else:
                alpha_cool = 1.0
            
            T = T0
            prev_x = x.copy()
            # Initial step direction (random)
            step_dir = rng.standard_normal(dim)
            
            for step in range(n_steps):
                if calls >= budget or time.time() >= deadline:
                    break
                
                # ASA update: x_{k+1} = x_k + step_size * (direction + random_perturbation)
                # The direction is based on the previous step to maintain momentum
                # Random perturbation magnitude is controlled by temperature
                
                # Update step direction with momentum and random component
                # The deterministic part maintains the search direction
                # The random part allows escaping local minima
                momentum = 0.5
                noise_scale = T
                step_dir = momentum * step_dir + noise_scale * rng.standard_normal(dim)
                
                # Normalize step direction to control step size
                step_norm = np.linalg.norm(step_dir)
                if step_norm > 1e-8:
                    step_dir = step_dir / step_norm
                else:
                    step_dir = rng.standard_normal(dim)
                    step_dir /= np.linalg.norm(step_dir)
                
                # Step size is proportional to temperature (larger when hot, smaller when cold)
                step_size = 0.5 * T
                if step_size < 1e-6:
                    step_size = 1e-6
                
                x_new = x + step_size * step_dir
                fx_new = try_call(x_new)
                if fx_new is None:
                    break
                
                # Accept or reject based on simulated annealing criterion
                if fx_new < fx:
                    # Always accept improvements
                    x = x_new
                    fx = fx_new
                    prev_x = x.copy()
                else:
                    # Accept with probability exp(-(fx_new - fx) / T)
                    delta_f = fx_new - fx
                    if delta_f > 0:
                        accept_prob = np.exp(-delta_f / max(T, 1e-10))
                        if rng.random() < accept_prob:
                            x = x_new
                            fx = fx_new
                            prev_x = x.copy()
                    else:
                        # Should not happen, but accept anyway
                        x = x_new
                        fx = fx_new
                        prev_x = x.copy()
                
                # Cool down
                T *= alpha_cool
                if T < T_end:
                    T = T_end

        # Phase 2: Local refinement using Nelder-Mead simplex method
        # This is crucial for the last few orders of magnitude of precision.
        if best_x is not None and calls < budget and time.time() < deadline:
            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                perturbation = best_x.copy()
                perturbation[i] += 0.5
                simplex.append(perturbation)
            
            simplex_f = [best_f]
            for i in range(1, dim + 1):
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    break
                simplex_f.append(fx_i)
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = simplex[i].copy()
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrinkage
            
            max_iterations = 200
            for _ in range(max_iterations):
                if calls >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by fitness
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Check convergence
                if len(set(simplex_f)) == 1 or (max(simplex_f) - min(simplex_f)) < 1e-10:
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break
                
                if f_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Try contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            f_i = try_call(simplex[i])
                            if f_i is None:
                                break
                            simplex_f[i] = f_i
                        if f_i is None:
                            break

        # Phase 3: Fine-tuning with small random perturbations around best
        # Use remaining budget for local exploration
        if best_x is not None and calls < budget and time.time() < deadline:
            # Start with small step size and gradually decrease
            step_size = 0.1
            no_improve = 0
            max_no_improve = 20
            
            while calls < budget and time.time() < deadline and no_improve < max_no_improve:
                y = best_x + step_size * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                    step_size *= 0.9  # Slightly reduce step size on improvement
                else:
                    no_improve += 1
                    step_size *= 0.8  # Reduce step size when no improvement
                    if step_size < 1e-6:
                        break

        return best_x, best_f
