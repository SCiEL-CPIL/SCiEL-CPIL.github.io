"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic Nelder-Mead simplex with 10 random initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        def safe_func(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val

        best_x = None
        best_f = np.inf
        
        # Phase 1: 10 random samples to find a good starting point
        n_init = 10
        for _ in range(n_init):
            if calls_made >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        def nelder_mead(x0, max_iter):
            nonlocal calls_made, best_f, best_x
            
            # Initialize simplex with smaller step for better initial convergence
            step = 0.3
            simplex = [x0.copy()]
            for i in range(dim):
                dx = x0.copy()
                dx[i] += step
                simplex.append(np.clip(dx, -5, 5))
            
            fvals = []
            for s in simplex:
                if calls_made >= total_budget:
                    break
                fvals.append(safe_func(s))
            
            if len(fvals) < dim + 1:
                return x0.copy(), best_f
            
            for _ in range(max_iter):
                if calls_made >= total_budget:
                    break
                    
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    
                # Stop criteria
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                center = np.mean(simplex, axis=0)
                size = np.mean([np.linalg.norm(s - center) for s in simplex])
                if size < 1e-6:
                    break

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)
                
                if calls_made >= total_budget:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + 2.0 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    if calls_made >= total_budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    if calls_made >= total_budget:
                        break
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= total_budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                        if calls_made >= total_budget:
                            break

            return simplex[0].copy(), fvals[0]

        # Phase 2: Run Nelder-Mead from the best initialization
        nelder_mead(best_x, max_iter=200)
        
        # Phase 3: If budget remains, do a few more random restarts with NM
        remaining = total_budget - calls_made
        if remaining > 50:
            n_restarts = min(3, remaining // 100)
            for _ in range(n_restarts):
                if calls_made >= total_budget - 30:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    if calls_made < total_budget - 30:
                        nelder_mead(best_x, max_iter=50)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
