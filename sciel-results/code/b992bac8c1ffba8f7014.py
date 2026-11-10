"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using multi-start local search.
    1. Stratified Latin Hypercube sampling for global initialization.
    2. Nelder-Mead simplex optimization from each anchor to exploit local regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        if budget <= 0:
            return None

        # Phase 1: Latin Hypercube Sampling
        # We want to cover the space well.
        # Let's use 40 anchors.
        n_anchors = min(40, budget)
        
        # Generate LHS points
        # Stratified sampling: divide each dimension into n_anchors equal intervals.
        # Randomly pick one point from each interval in each dimension.
        # To ensure distinct points, we shuffle the assignments.
        
        rng = np.random.default_rng(42)
        
        # For each dimension, create n_anchors bins.
        # For dim d, bin i covers [lo[d] + i*(hi[d]-lo[d])/n, lo[d] + (i+1)*(hi[d]-lo[d])/n]
        # We pick a random offset in each bin.
        # To make it a Latin Hypercube, we ensure that each "row" (anchor) uses a unique "column" (bin index) per dimension.
        
        points = np.zeros((n_anchors, dim))
        
        for d in range(dim):
            # Create a permutation of 0..n_anchors-1
            perm = rng.permutation(n_anchors)
            # For each anchor i, it falls into bin perm[i]
            # The point within the bin is random
            # bin_width = (hi[d] - lo[d]) / n_anchors
            # point = lo[d] + (perm[i] + rng.random()) * bin_width
            
            bin_width = (hi[d] - lo[d]) / n_anchors
            offsets = rng.random(n_anchors)
            points[:, d] = lo[d] + (perm + offsets) * bin_width
        
        # Evaluate all anchor points
        F_anchors = np.zeros((n_anchors, n_obj))
        evals_used = 0
        for i in range(n_anchors):
            if evals_used >= budget:
                break
            F_anchors[i] = problem(points[i])
            evals_used += 1
            if time.time() > deadline:
                break
        
        # Phase 2: Nelder-Mead from each anchor
        # We have remaining = budget - evals_used evaluations.
        # We want to distribute these among the anchors.
        # Let's do a fair share: each anchor gets remaining // n_anchors evaluations.
        # Plus we might do some random restarts if time permits? No, stick to the plan.
        
        # Nelder-Mead implementation
        # We need to minimize a scalar function.
        # For multi-objective, we can minimize the Tchebycheff distance to the ideal point,
        # or simply minimize the sum of objectives, or use a random weight vector.
        # A common strategy for multi-objective local search is to minimize a scalarized objective
        # with random weights to explore different parts of the front.
        
        # However, the directive says "independent, budget-constrained Nelder-Mead simplex optimizations from each anchor".
        # It doesn't specify the scalarization. Let's use a random weight vector for each anchor
        # to diversify the search directions. Or better, use the "ideal point" approximation.
        
        # Let's try minimizing the Euclidean distance to the current best ideal point (min of seen objectives).
        # Or simply: for each anchor, pick a random weight vector w > 0, and minimize sum(w_i * f_i(x)).
        # This is standard weighted sum method.
        
        # Let's refine:
        # For each anchor i:
        #   1. Generate a random weight vector w_i.
        #   2. Run Nelder-Mead to minimize sum(w_i * f(x)) starting from x_i.
        #   3. Limit evaluations to remaining_budget // n_anchors.
        
        remaining_budget = budget - evals_used
        if remaining_budget <= 0:
            return None
        
        evals_per_anchor = max(1, remaining_budget // n_anchors)
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        # We need to be careful with the simplex size.
        # Let's initialize the simplex around the anchor.
        # Initial step size: 0.1 * (hi - lo)
        
        for i in range(n_anchors):
            if evals_used >= budget:
                break
            if time.time() > deadline:
                break
            
            x_start = points[i].copy()
            
            # Random weights
            w = rng.random(n_obj) + 0.1 # Ensure positive
            
            def scalarized_obj(x, w=w):
                f = problem(x)
                evals_used_local[0] += 1 # We need to track this carefully
                return np.dot(w, f)
            
            # We need a way to count evaluations inside the Nelder-Mead loop without
            # making it too complex.
            # Let's implement a simple Nelder-Mead that takes a function and a budget.
            
            # Simple Nelder-Mead implementation
            # simplex: list of (dim+1) points
            # We start with x_start and perturbations.
            
            simplex = [x_start.copy()]
            for d in range(dim):
                x_pert = x_start.copy()
                # Perturb in dimension d
                # If x_start[d] is close to boundary, perturb inwards?
                # Just use a small step.
                step = 0.1 * (hi[d] - lo[d])
                x_pert[d] += step
                x_pert = np.clip(x_pert, lo, hi)
                simplex.append(x_pert)
            
            # Evaluate initial simplex
            f_simplex = np.zeros(dim + 1)
            for j in range(dim + 1):
                if evals_used >= budget:
                    break
                f_simplex[j] = np.dot(w, problem(simplex[j]))
                evals_used += 1
                if time.time() > deadline:
                    break
            
            # If we broke out early due to budget, skip optimization for this anchor
            if evals_used >= budget or time.time() > deadline:
                break
                
            # Main Nelder-Mead loop
            # We have evals_per_anchor evaluations left for this anchor.
            # Actually, we already used dim+1.
            # Let's say we have evals_per_anchor - (dim+1) left?
            # Or just use a total budget of evals_per_anchor for the whole NM process including initialization?
            # The directive says "allocates the remaining 160 evaluations to perform... optimizations from each anchor".
            # So each anchor gets a slice of the remaining budget.
            # Let's say we have `local_budget` evaluations left for this anchor.
            # local_budget = evals_per_anchor
            # We already used dim+1.
            # So loop while evals_used < budget and local_evals_used < local_budget
            
            local_evals_used = dim + 1
            local_budget = evals_per_anchor
            
            while local_evals_used < local_budget and evals_used < budget:
                if time.time() > deadline:
                    break
                    
                # Sort simplex by objective value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = f_simplex[order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = np.dot(w, problem(x_r))
                evals_used += 1
                local_evals_used += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = np.dot(w, problem(x_e))
                    evals_used += 1
                    local_evals_used += 1
                    
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
                    # Contraction
                    # Outside or inside contraction?
                    # If f_r > f_simplex[-1], outside contraction.
                    # If f_r < f_simplex[-1], inside contraction.
                    # Standard: if f_r > f_simplex[-1], outside.
                    if f_r > f_simplex[-1]:
                        x_c = centroid + rho * (simplex[-1] - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                    
                    x_c = np.clip(x_c, lo, hi)
                    f_c = np.dot(w, problem(x_c))
                    evals_used += 1
                    local_evals_used += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            f_simplex[j] = np.dot(w, problem(simplex[j]))
                            evals_used += 1
                            local_evals_used += 1
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break

        return None

# EVOLVE-BLOCK-END
