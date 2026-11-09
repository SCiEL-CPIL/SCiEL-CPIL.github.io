"""Combined optimizer: Halton-based multi-start (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline - 5.0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sampling using Halton sequences
        # Halton provides better space-filling properties than random or grid
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_init = 15
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        init_points = []
        # Add center
        init_points.append(np.zeros(dim))
        
        # Generate Halton points
        for i in range(1, n_init):
            p = np.array([halton(i + 1, primes[j]) * 10 - 5 for j in range(dim)])
            init_points.append(p)

        # Evaluate initial points
        init_results = []
        for p in init_points:
            if call_count >= budget - 50 or check_time():
                break
            f = evaluate(p)
            init_results.append((p, f))

        if not init_results:
            return best_x, best_f

        # Sort initial results by fitness
        init_results.sort(key=lambda x: x[1])

        # Phase 2: Multi-start (1+1)-ES on top candidates
        # Allocate ~50% of remaining budget to (1+1)-ES
        remaining = budget - call_count
        es_budget = int(remaining * 0.5)
        nm_budget = remaining - es_budget

        # Select top candidates for ES
        n_es_starts = min(5, len(init_results))
        
        # Distribute ES budget: give more to better candidates
        weights = np.array([1.0, 0.6, 0.4, 0.25, 0.15][:n_es_starts])
        weights /= weights.sum()
        es_allocs = (weights * es_budget).astype(int)
        # Ensure at least 10 calls per start if possible
        es_allocs = np.maximum(es_allocs, 10)

        for i in range(n_es_starts):
            if call_count >= budget - 10 or check_time():
                break
            
            x0, f0 = init_results[i]
            alloc = es_allocs[i]
            
            # Run (1+1)-ES
            x = x0.copy()
            f = f0
            sigma = 0.5  # Initial step size
            successes = 0
            attempts = 0
            window = 10

            for _ in range(alloc):
                if call_count >= budget - 10 or check_time():
                    break
                
                # Generate trial point
                d = rng.standard_normal(dim)
                x_trial = x + sigma * d
                f_trial = evaluate(x_trial)
                
                if f_trial < f:
                    x = x_trial.copy()
                    f = f_trial
                    successes += 1
                attempts += 1
                
                # Adapt sigma
                if attempts >= window:
                    rate = successes / attempts
                    if rate > 0.5:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                    attempts = 0

        # Phase 3: Nelder-Mead Polish on the global best
        if best_x is not None and call_count < budget - 5 and not check_time():
            # Initialize simplex
            simplex = [best_x.copy()]
            f_simplex = [best_f]
            
            step = 0.2
            for i in range(dim):
                x_new = best_x.copy()
                x_new[i] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = evaluate(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
                
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            max_nm_calls = budget - call_count
            for _ in range(max_nm_calls):
                if call_count >= budget or check_time():
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                
                if f_r < f_simplex[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Try contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_simplex[i] = evaluate(simplex[i])
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
