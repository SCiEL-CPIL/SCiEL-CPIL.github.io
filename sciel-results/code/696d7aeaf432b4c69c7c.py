"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Dense LHS global search + Nelder-Mead local exploitation for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

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

        # Phase 1: Dense Latin Hypercube Sampling to find the best basin
        # We use a stratified grid approach for efficiency
        # For dim=5, we can do a reasonable LHS with ~100-200 points
        
        rng = np.random.default_rng(12345)
        
        # Use a structured approach: evaluate points on a coarse grid + LHS
        # First, evaluate center and some structured points
        n_lhs = 150  # number of LHS samples
        
        # Generate LHS points
        # For each dimension, divide [-4, 4] into n_lhs strata
        # and pick one random point per stratum
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create strata boundaries
            edges = np.linspace(-4.0, 4.0, n_lhs + 1)
            # Pick random point in each stratum
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        
        # Shuffle to decorrelate
        perm = rng.permutation(n_lhs)
        lhs_points = lhs_points[perm]
        
        # Evaluate all LHS points
        for i in range(n_lhs):
            if calls_made >= budget - 50 or time.time() > deadline - 10:
                break
            evaluate(lhs_points[i])
        
        # Phase 2: Nelder-Mead local optimization from the best point
        if global_best is not None and calls_made < budget - 20 and time.time() < deadline - 10:
            x = global_best.copy()
            fx = global_fx
            
            # Nelder-Mead simplex
            # Initial simplex: centered at x, with step size based on problem scale
            step = 1.0  # initial step size
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x
            for i in range(1, dim + 1):
                simplex[i] = x.copy()
                simplex[i, i-1] += step
            
            # Evaluate simplex vertices
            f_simplex = np.zeros(dim + 1)
            f_simplex[0] = fx
            for i in range(1, dim + 1):
                if calls_made >= budget - 10 or time.time() > deadline - 5:
                    break
                f_simplex[i] = evaluate(simplex[i])
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink
            max_iter_nm = 200
            
            for it in range(max_iter_nm):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Check convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = evaluate(x_ref)
                
                if f_ref < f_simplex[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_simplex[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, f_simplex[-1]):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made >= budget - 5 or time.time() > deadline - 5:
                                break
                            f_simplex[i] = evaluate(simplex[i])
            
            # After Nelder-Mead, do a final local search with decreasing step
            if global_best is not None and calls_made < budget - 5 and time.time() < deadline - 5:
                x = global_best.copy()
                fx = global_fx
                step_local = 0.5
                for _ in range(10):
                    if calls_made >= budget - 2 or time.time() > deadline - 2:
                        break
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            y = x.copy()
                            y[d] += direction * step_local
                            y = np.clip(y, -5.0, 5.0)
                            fy = evaluate(y)
                            if fy < fx:
                                x = y
                                fx = fy
                                improved = True
                    if not improved:
                        step_local *= 0.5
                    else:
                        step_local *= 0.8

        return global_best, global_fx
# EVOLVE-BLOCK-END
