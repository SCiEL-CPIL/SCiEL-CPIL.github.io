"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with global random search for continuous black-box problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = [0]

        def wrapped(x):
            if call_count[0] >= budget:
                return float('inf')
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            if not np.isfinite(val):
                return 1e30
            return float(val)

        best_x = None
        best_f = float('inf')

        def nelder_mead(x0, max_calls):
            nonlocal call_count, best_f, best_x
            
            if call_count[0] >= budget:
                return x0, wrapped(x0)
                
            # Initialize simplex
            simplex = []
            f_values = []
            
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped(x0)
            simplex.append(x0.copy())
            f_values.append(f0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Create other points
            for i in range(dim):
                x_new = x0.copy()
                if x0[i] == 0:
                    step = 0.25
                else:
                    step = 0.05 * abs(x0[i])
                x_new[i] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = wrapped(x_new)
                simplex.append(x_new)
                f_values.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    
            while call_count[0] < budget:
                # Sort simplex
                idx = np.argsort(f_values)
                simplex = [simplex[i] for i in idx]
                f_values = [f_values[i] for i in idx]
                
                # Check convergence
                f_min = f_values[0]
                f_max = f_values[-1]
                if abs(f_max - f_min) < 1e-10:
                    break
                    
                x_worst = simplex[-1]
                x_best = simplex[0]
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = x_centroid + alpha * (x_centroid - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = wrapped(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = x_centroid + gamma * (x_centroid - x_worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = wrapped(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                else:
                    # Contraction
                    beta = 0.5
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = x_centroid + beta * (x_r - x_centroid)
                    else:
                        # Inside contraction
                        x_c = x_centroid - beta * (x_centroid - x_worst)
                        
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = wrapped(x_c)
                    
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = wrapped(simplex[i])
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                            
                            if call_count[0] >= budget:
                                break
                        if call_count[0] >= budget:
                            break

            return best_x, best_f

        # Strategy:
        # 1. Global search with random points (10% budget)
        # 2. Local search with Nelder-Mead (90% budget)
        
        num_global = max(10, budget // 10)
        
        # Global Search
        for _ in range(num_global):
            if call_count[0] >= budget:
                break
            x_rand = rng.uniform(-5.0, 5.0, dim)
            f_rand = wrapped(x_rand)
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()
                
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
            
        # Local Search with Nelder-Mead
        remaining = budget - call_count[0]
        if remaining > 0:
            nelder_mead(best_x, remaining)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
