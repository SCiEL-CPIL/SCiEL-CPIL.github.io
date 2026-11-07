"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for Lunacek bi-Rastrigin (f24).
    Phase 1: Dense Latin Hypercube Sampling for global basin identification.
    Phase 2: Nelder-Mead simplex method for local exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        # --- Phase 1: Global Search with Dense LHS ---
        # Use 30% of budget for global search to ensure basin identification
        n_global = int(0.3 * budget)
        n_global = max(n_global, 50)
        n_global = min(n_global, budget - 10)  # Leave some for local search
        
        rng = np.random.default_rng(42)
        
        # LHS points in [-4.5, 4.5]^dim
        # The optimum is guaranteed to be in [-4, 4]^5
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            edges = np.linspace(-4.5, 4.5, n_global + 1)
            for i in range(n_global):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        
        # Shuffle to ensure diversity in evaluation order
        perm = rng.permutation(n_global)
        lhs_points = lhs_points[perm]

        for i in range(n_global):
            if calls_made >= budget - 10 or time.time() > deadline - 10:
                break
            evaluate(lhs_points[i])

        # --- Phase 2: Local Search with Nelder-Mead ---
        if global_best is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            x0 = global_best.copy()
            
            # Nelder-Mead parameters
            # Initial simplex: centered at x0 with small spread
            spread = 0.5
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, i-1] += spread
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                simplex_f[i] = evaluate(simplex[i])
            
            # Nelder-Mead constants
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            # Main Nelder-Mead loop
            max_iters = 1000
            for it in range(max_iters):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10 * (1 + abs(np.min(simplex_f))):
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = evaluate(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    f_con = evaluate(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            if calls_made >= budget - 5 or time.time() > deadline - 5:
                                break
                        for i in range(1, dim + 1):
                            if calls_made >= budget - 5 or time.time() > deadline - 5:
                                break
                            simplex_f[i] = evaluate(simplex[i])
            
            # Final check: ensure global_best is the best found
            if global_best is not None:
                # One more evaluation at the best point to be safe
                pass  # Already tracked

        return global_best, global_fx
# EVOLVE-BLOCK-END
