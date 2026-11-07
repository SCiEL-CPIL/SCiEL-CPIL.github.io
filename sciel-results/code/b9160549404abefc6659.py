"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Latin Hypercube sampling + Nelder-Mead hybrid optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0
        global_x = None
        global_fx = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls_used, global_x, global_fx
            if calls_used >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for global basin identification
        # Use ~40% of budget for global exploration
        lhs_budget = int(budget * 0.4)
        n_lhs = min(lhs_budget, 200)
        
        rng = np.random.default_rng(42)
        
        # Generate Latin Hypercube sample
        # For each dimension, divide [−5, 5] into n_lhs equal intervals
        # and sample one point from each interval
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Permutation for each dimension
            perm = rng.permutation(n_lhs)
            # Uniform sample within each interval
            offsets = rng.uniform(0, 1, n_lhs)
            lhs_points[:, d] = -5 + (perm + offsets) * (10.0 / n_lhs)
        
        # Evaluate all LHS points
        for i in range(n_lhs):
            if calls_used >= budget or not time_ok():
                break
            evaluate(lhs_points[i])
        
        # Phase 2: Nelder-Mead local optimization from best LHS point
        if global_x is not None and calls_used < budget and time_ok():
            # Create a copy to avoid modifying global_x during NM
            nm_start = global_x.copy()
            
            # Nelder-Mead with adaptive step size
            # Initial simplex: use a moderate step size
            step = 1.0
            
            # Create initial simplex
            simplex = [nm_start.copy()]
            for d in range(dim):
                pt = nm_start.copy()
                pt[d] += step
                simplex.append(pt)
            
            # Evaluate initial simplex
            simplex_f = [np.inf] * (dim + 1)
            for i in range(dim + 1):
                if calls_used >= budget or not time_ok():
                    break
                simplex_f[i] = evaluate(simplex[i])
            
            # Nelder-Mead iterations
            max_nm_iter = 50
            for it in range(max_nm_iter):
                if calls_used >= budget or not time_ok():
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                f_reflect = evaluate(x_reflect)
                
                if f_reflect < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    f_expand = evaluate(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    f_contract = evaluate(x_contract)
                    if f_contract < simplex_f[-1]:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
            
            # Update global best after NM
            if global_x is not None:
                # Final evaluation to ensure we have the best
                pass

        # Phase 3: Polish with L-BFGS-B if budget remains
        if global_x is not None and calls_used < budget and time_ok():
            polish_budget = budget - calls_used
            if polish_budget > 10:
                try:
                    def polished_func(z):
                        nonlocal calls_used
                        if calls_used >= budget:
                            return func(z)
                        calls_used += 1
                        return func(z)
                    
                    res = scipy_minimize(polished_func, global_x, method='L-BFGS-B',
                                        bounds=[(-5, 5)] * dim,
                                        options={'maxiter': min(polish_budget, 50),
                                                 'maxfun': min(polish_budget, 50),
                                                 'gtol': 1e-8})
                    if res.fun < global_fx:
                        global_fx = res.fun
                        global_x = res.x.copy()
                except Exception:
                    pass

        return global_x, global_fx
# EVOLVE-BLOCK-END
