"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Dense LHS + Fixed-Step Nelder-Mead."""
import numpy as np
import time

class Optimizer:
    """
    Hybrid Dense LHS + Fixed-Step Nelder-Mead optimizer for Lunacek bi-Rastrigin.
    
    Strategy:
    1. Global Phase: Use a dense deterministic Latin Hypercube (LHS) to sample the space.
       LHS ensures at least one point per stratum in each dimension, providing robust
       coverage to identify the global basin vs the shallow basin.
       We allocate 30% of the budget (150 points) to this phase.
    2. Local Phase: Use a standard Nelder-Mead simplex with a fixed initial step size.
       We avoid adaptive step logic to reduce overhead and potential instability,
       relying on the dense global search to provide a good starting point.
       The simplex is initialized with a step size of 0.5 (half the width of the LHS strata
       in a 300-point 5D grid, roughly) to balance exploration and exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        # 300s limit per task contract, so 290s deadline is safe.
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Call counter
        calls_ref = [0]
        
        def nelder_mead(x0, f0, max_calls, func, calls_ref):
            """
            Standard Nelder-Mead.
            - Initializes simplex with a fixed step size.
            - Uses standard reflection, expansion, contraction, shrinkage.
            """
            n = dim
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Initial step size: 0.5 is a reasonable choice for [-5, 5]^5
            # It's small enough for local convergence but large enough to handle noise/flat areas
            step = 0.5
            
            for i in range(n):
                new_x = x0.copy()
                new_x[i] += step
                new_x = np.clip(new_x, -5.0, 5.0)
                simplex.append(new_x)
                f_val = func(new_x)
                calls_ref[0] += 1
                f_values.append(f_val)
                
                if calls_ref[0] >= max_calls:
                    return x0, f0

            best_x = x0.copy()
            best_f = f0
            
            while calls_ref[0] < max_calls and time_ok():
                # Sort simplex
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Update best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                calls_ref[0] += 1
                
                if calls_ref[0] >= max_calls:
                    break

                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    calls_ref[0] += 1
                    
                    if calls_ref[0] >= max_calls:
                        break

                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func(xc)
                    calls_ref[0] += 1
                    
                    if calls_ref[0] >= max_calls:
                        break

                    if fc < f_values[-1]:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = func(simplex[i])
                            calls_ref[0] += 1
                            if calls_ref[0] >= max_calls:
                                return best_x, best_f
                            
            return best_x, best_f

        # Phase 1: Global Exploration with Dense Latin Hypercube
        # We use 30% of the budget for global search to ensure we find the global basin.
        # 150 points in 5D is dense enough to distinguish the two basins of Lunacek.
        n_global = int(0.3 * budget)
        if n_global < 10:
            n_global = 10
            
        # Generate LHS samples in [0, 1]^dim
        # Standard LHS: partition each dimension into n_global strata, permute, and scale.
        # This is deterministic if we use a fixed permutation seed, but here we can use
        # a simple deterministic construction or a fixed random seed for reproducibility
        # across instances (though instances have hidden shifts, the LHS structure remains
        # valid for coverage).
        # We use a fixed seed for the permutation to ensure consistent behavior.
        rng = np.random.RandomState(42)
        
        l_samples = np.zeros((n_global, dim))
        for d in range(dim):
            # Create strata centers
            strata_centers = (np.arange(n_global) + 0.5) / n_global
            # Random permutation for LHS
            perm = rng.permutation(n_global)
            l_samples[:, d] = strata_centers[perm]
        
        # Map [0, 1] to [-5, 5]
        l_samples = l_samples * 10.0 - 5.0
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Evaluate global samples
        for i in range(n_global):
            if not time_ok() or calls_ref[0] >= budget:
                break
            x = l_samples[i]
            f = func(x)
            calls_ref[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local Refinement with Fixed-Step Nelder-Mead
        if calls_ref[0] < budget and time_ok():
            x0 = best_x.copy()
            f0 = best_f
            # Allocate remaining budget to NM
            max_nm_calls = budget - calls_ref[0]
            best_x, best_f = nelder_mead(x0, f0, max_nm_calls, func, calls_ref)
            
        # If we have time and budget left, we could try a second local run from the
        # second-best global point if the first one didn't converge well, but for
        # simplicity and robustness, one strong local run from the best global point
        # is usually sufficient.
        
        return best_x, best_f
