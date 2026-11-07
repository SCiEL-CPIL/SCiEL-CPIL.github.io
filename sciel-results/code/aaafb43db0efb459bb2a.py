"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES exploration + multi-start Nelder-Mead refinement with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
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

        rng = np.random.default_rng(42)

        # Phase 1: (1+1)-ES for global exploration
        # Use a fraction of budget for exploration to find a good starting point
        es_budget = min(80, budget // 5)

        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        calls += 1
        record(x, fx)
        sigma = 2.0  # Start with larger step for exploration

        for _ in range(es_budget - 1):
            if not time_ok() or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                record(x, fy)
            else:
                sigma *= 0.9
            if not time_ok():
                break

        # Phase 2: Multi-start Nelder-Mead local refinement
        if not time_ok() or calls >= budget or best_x is None:
            return best_x, best_f

        # Generate diverse starting points for NM refinement
        # 1. The best point found by ES
        # 2. A few random points around the best point
        # 3. A few random points in the box
        nm_starts = [best_x.copy()]
        
        # Add points near the best solution
        for _ in range(2):
            x_near = best_x + 0.5 * rng.standard_normal(dim)
            x_near = np.clip(x_near, -5, 5)
            nm_starts.append(x_near)
        
        # Add random points in the box for diversity
        for _ in range(2):
            x_rand = rng.uniform(-4, 4, dim)
            nm_starts.append(x_rand)

        # Evaluate all NM starting points (except the best which we already have)
        evaluated_starts = [(best_x.copy(), best_f)]
        for x0 in nm_starts[1:]:
            if not time_ok() or calls >= budget:
                break
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            evaluated_starts.append((x0.copy(), f0))
        
        if not evaluated_starts:
            return best_x, best_f

        # Sort starting points by function value
        evaluated_starts.sort(key=lambda t: t[1])
        
        # Run Nelder-Mead from the best few starting points
        # We'll run Nelder-Mead on up to 3 of the best starting points
        nm_starts_to_refine = evaluated_starts[:min(3, len(evaluated_starts))]
        
        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        alpha = 0.5  # Initial simplex size

        for x0, f0 in nm_starts_to_refine:
            if not time_ok() or calls >= budget:
                break
            
            # Initialize simplex around this starting point
            simplex = [x0.copy()]
            f_simplex = [f0]
            
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
            
            # Run Nelder-Mead iterations
            max_nm_iters = 100
            for _ in range(max_nm_iters):
                if not time_ok() or calls >= budget:
                    break
                
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
