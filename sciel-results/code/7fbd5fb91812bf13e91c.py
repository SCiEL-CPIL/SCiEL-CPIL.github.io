import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        best_x_list = []
        best_f_list = []

        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            if eval_count >= budget:
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        # 1. Initialization: Structured grid + random
        # Rastrigin has local minima at integers. The global min is at the shifted origin.
        # Since we don't know the shift, we sample integers and near-integers.
        # Also sample the origin (0,0,0,0,0) as it's a common candidate for "unshifted" or small shifts.
        
        rng = np.random.default_rng(12345)
        
        init_points = []
        
        # Origin
        init_points.append(np.zeros(dim))
        
        # Integer grid points (subset)
        # Range [-4, 4] for integers is 9^5 = 59049 which is too many.
        # Let's pick a sparse grid or random integers.
        # Actually, let's just do a few random integers and random points.
        # The shift is hidden but in [-4, 4].
        
        # Sample random points in [-5, 5]
        for _ in range(10):
            init_points.append(rng.uniform(-5, 5, dim))
            
        # Sample random integer points
        for _ in range(10):
            p = rng.integers(-4, 5, size=dim).astype(float)
            init_points.append(p)
            
        # Sample points near integers (jitter)
        for _ in range(10):
            p = rng.integers(-4, 5, size=dim).astype(float)
            p += rng.uniform(-0.5, 0.5, dim)
            p = np.clip(p, -5, 5)
            init_points.append(p)

        # Evaluate initial points
        for p in init_points:
            if eval_count >= budget or time_check():
                break
            wrapped_func(p)

        # 2. Multi-start CMA-ES
        # We will run several independent CMA-ES instances.
        # Each instance gets a portion of the budget.
        
        # Strategy:
        # - Pick top 3 distinct starting points from initialization (or random if not enough)
        # - Run CMA-ES from each, allocating remaining budget / 3 to each.
        # - Keep the global best.
        
        # Sort init points by fitness to find good starts
        # We didn't store them, so let's re-evaluate a small set or just use random restarts from the best found so far + random.
        # Better: Just use the current best_x as one start, and generate 2 random starts near it or in the box.
        
        # Let's implement a robust CMA-ES class inside to avoid state pollution.
        
        def cma_es(start_x, max_evals, seed):
            nonlocal eval_count, best_x, best_f
            
            if eval_count >= budget or time_check():
                return
            
            # CMA-ES Parameters
            lam = 10
            mu = lam // 2
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            x_mean = start_x.copy()
            # Initial sigma: 1.0 is reasonable for [-5,5] box if start is somewhat central
            sigma = 1.0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
            c_hs = 1.1 + 2.0 / (dim + 1.7)
            
            generation = 0
            local_rng = np.random.default_rng(seed)
            
            while eval_count < budget and not time_check():
                # Check if we have enough budget for a full generation
                if eval_count + lam > budget:
                    # Do a partial generation? Or just break to save for local search?
                    # Let's just break to avoid wasting evaluations on incomplete updates
                    break
                
                # 1. Sample
                ys = []
                for k in range(lam):
                    z = local_rng.standard_normal(dim)
                    y = B @ (D * z)
                    ys.append(y)
                
                xs = x_mean + sigma * np.array(ys)
                
                # 2. Evaluate
                fitnesses = []
                eval_xs = []
                for i in range(lam):
                    if eval_count >= budget or time_check():
                        break
                    x = xs[i]
                    f = wrapped_func(x)
                    fitnesses.append(f)
                    eval_xs.append(x)
                
                if len(fitnesses) < lam:
                    break # Incomplete generation, stop CMA-ES
                
                fitnesses = np.array(fitnesses)
                eval_xs = np.array(eval_xs)
                
                # 3. Sort
                order = np.argsort(fitnesses)
                xs_sorted = eval_xs[order]
                ys_array = np.array(ys)
                ys_sorted = ys_array[order][:mu]
                
                # 4. Update mean
                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # 5. Update sigma
                y_mean = (new_mean - x_mean) / sigma
                
                norm_y_mean = np.linalg.norm(y_mean)
                sqrt_trace_C = np.sqrt(np.trace(C))
                ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
                
                p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                gen = generation + 1
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1.4 + 2 / (dim + 1)) * chi_n)
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                generation = gen
                
                # 6. Update C
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                rank_one = np.outer(p_c, p_c)
                if not h_sigma:
                    rank_one += c_c * (2 - c_c) * C
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                x_mean = new_mean
                
                # 7. Decompose
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.clip(D2, 1e-10, None)
                    D = np.sqrt(D2)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    
                # Stagnation
                if sigma < 1e-5:
                    break

        # Run 3 CMA-ES instances
        # Start 1: Current best
        # Start 2: Random point
        # Start 3: Random point
        
        remaining_budget = budget - eval_count
        if remaining_budget > 0 and not time_check():
            evals_per_run = remaining_budget // 3
            if evals_per_run < 20:
                evals_per_run = remaining_budget
            
            # Start 1
            cma_es(best_x, evals_per_run, 100)
            
            # Start 2
            if eval_count < budget and not time_check():
                start2 = rng.uniform(-5, 5, dim)
                cma_es(start2, evals_per_run, 200)
            
            # Start 3
            if eval_count < budget and not time_check():
                start3 = rng.uniform(-5, 5, dim)
                cma_es(start3, evals_per_run, 300)

        # 3. Local Refinement (Nelder-Mead or Coordinate Descent)
        # Use remaining budget for local search on the global best.
        
        if eval_count < budget and not time_check():
            local_x = best_x.copy()
            local_f = best_f
            
            # Simple Coordinate Descent with adaptive step
            step = 0.1
            max_local_evals = min(50, budget - eval_count)
            
            for _ in range(max_local_evals):
                if eval_count >= budget or time_check():
                    break
                
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        trial_x = local_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, -5, 5)
                        
                        if eval_count >= budget or time_check():
                            break
                        
                        trial_f = wrapped_func(trial_x)
                        if trial_f < local_f:
                            local_x = trial_x
                            local_f = trial_f
                            improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
