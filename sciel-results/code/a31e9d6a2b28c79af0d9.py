"""Multi-objective optimizer for ZDT1 using an adaptive hypervolume-based evolutionary algorithm."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-based multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, budget // 5)
        if pop_size < 10:
            pop_size = 10

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Initialize population with random points
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (both are objective vectors)."""
            return np.all(a <= b) and np.any(a < b)

        def hv_contribution(point, pop, ref):
            """Calculate hypervolume contribution of a point in the population.
            
            For 2 objectives, this is the area of the region in objective space
            that would be lost if the point were removed.
            """
            # Sort population by first objective
            indices = np.argsort(pop[:, 0])
            sorted_pop = pop[indices]
            
            # Find the point's position
            # We need to find the region unique to this point
            # For point p, the contribution is the area of the rectangle
            # [p[0], max(p[0], next_p[0])] x [p[1], min(p[1], ref[1])]
            # that is not covered by any other point
            
            # More robust approach: calculate HV of pop with and without the point
            # But that's expensive. Let's use the standard 2D formula.
            
            # For 2D, the HV contribution of point i is:
            # (max(p_i[0], p_{i+1}[0]) - p_i[0]) * (ref[1] - p_i[1])
            # where p_{i+1} is the next point in sorted order by first objective
            # and we take min with ref[0]
            
            # Actually, let's just compute HV of the set with and without the point
            # for simplicity, since pop_size is small
            mask = np.ones(len(pop), dtype=bool)
            # Find which index corresponds to the point
            # This is tricky if there are duplicates. Let's assume unique points.
            # Better: pass the index
            
            pass  # We'll use a different approach below

        def hypervolume_2d(points, ref):
            """Calculate hypervolume for 2D points with given reference point.
            
            Uses the standard sweep line algorithm for 2D.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than reference
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            points = points[valid]
            
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            points = points[np.argsort(points[:, 0])]
            
            # Sweep line: keep track of minimum second objective
            hv = 0.0
            min_second = ref[1]
            
            for i in range(len(points) - 1, -1, -1):
                p = points[i]
                if p[1] < min_second:
                    hv += (points[i + 1][0] - p[0]) * (min_second - p[1]) if i + 1 < len(points) else (ref[0] - p[0]) * (min_second - p[1])
                    min_second = p[1]
            
            return hv

        def hv_contribution_2d(point, pop, ref):
            """Calculate the hypervolume contribution of a specific point in the population."""
            # HV with point
            hv_with = hypervolume_2d(pop, ref)
            
            # HV without point
            mask = np.ones(len(pop), dtype=bool)
            # Find the point in pop
            # Since we're passing the actual point, find its index
            for i in range(len(pop)):
                if np.allclose(pop[i], point):
                    mask[i] = False
                    break
            
            pop_without = pop[mask]
            hv_without = hypervolume_2d(pop_without, ref)
            
            return hv_with - hv_without

        def non_dominated_sort(pop):
            """Return indices of non-dominated points in the population."""
            n = len(pop)
            dominated = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(pop[j], pop[i]):
                        dominated[i] += 1
            return np.where(dominated == 0)[0]

        # Main loop
        while used < budget:
            # Get non-dominated points
            nondom_idx = non_dominated_sort(F)
            
            if len(nondom_idx) == 0:
                # Fallback: select random parents
                parent1_idx = np.random.randint(pop_size)
                parent2_idx = np.random.randint(pop_size)
            else:
                # Select parents using tournament selection based on hypervolume contribution
                # Calculate HV contributions for non-dominated points
                contributions = np.zeros(len(nondom_idx))
                for i, idx in enumerate(nondom_idx):
                    contributions[i] = hv_contribution_2d(F[idx], F[nondom_idx], ref)
                
                # Tournament selection: pick 2 random from non-dominated, keep better
                if len(nondom_idx) > 1:
                    t1 = np.random.randint(len(nondom_idx))
                    t2 = np.random.randint(len(nondom_idx))
                    if contributions[t1] >= contributions[t2]:
                        parent1_idx = nondom_idx[t1]
                    else:
                        parent1_idx = nondom_idx[t2]
                    
                    t1 = np.random.randint(len(nondom_idx))
                    t2 = np.random.randint(len(nondom_idx))
                    if contributions[t1] >= contributions[t2]:
                        parent2_idx = nondom_idx[t1]
                    else:
                        parent2_idx = nondom_idx[t2]
                else:
                    parent1_idx = nondom_idx[0]
                    parent2_idx = nondom_idx[0]
            
            # Simulated Binary Crossover (SBX)
            # For each dimension, with probability 0.5, apply SBX
            child1 = X[parent1_idx].copy()
            child2 = X[parent2_idx].copy()
            
            eta_c = 15.0
            for d in range(dim):
                if np.random.random() < 0.5:
                    # SBX
                    x1 = X[parent1_idx, d]
                    x2 = X[parent2_idx, d]
                    if abs(x1 - x2) > 1e-14:
                        if np.random.random() < 0.5:
                            beta = 1.0 + 2.0 * (x1 - lo[d]) / (x2 - x1)
                            alpha = 2.0 - beta ** (-(eta_c + 1.0))
                            if np.random.random() <= 1.0 / alpha:
                                beta_q = beta ** (-(1.0 / (eta_c + 1.0)))
                            else:
                                beta_q = (1.0 / (2.0 - 2.0 * beta)) ** (-(1.0 / (eta_c + 1.0)))
                            child1[d] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                        else:
                            beta = 1.0 + 2.0 * (x2 - lo[d]) / (x1 - x2)
                            alpha = 2.0 - beta ** (-(eta_c + 1.0))
                            if np.random.random() <= 1.0 / alpha:
                                beta_q = beta ** (-(1.0 / (eta_c + 1.0)))
                            else:
                                beta_q = (1.0 / (2.0 - 2.0 * beta)) ** (-(1.0 / (eta_c + 1.0)))
                            child2[d] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
            
            # Polynomial Mutation
            eta_m = 20.0
            for d in range(dim):
                if np.random.random() < 1.0 / dim:
                    delta1 = (child1[d] - lo[d]) / (hi[d] - lo[d])
                    delta2 = (hi[d] - child1[d]) / (hi[d] - lo[d])
                    rand = np.random.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1.0)
                        delta_q = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1.0)
                        delta_q = 1.0 - (val ** (1.0 / (eta_m + 1.0)))
                    child1[d] += delta_q * (hi[d] - lo[d])
                
                if np.random.random() < 1.0 / dim:
                    delta1 = (child2[d] - lo[d]) / (hi[d] - lo[d])
                    delta2 = (hi[d] - child2[d]) / (hi[d] - lo[d])
                    rand = np.random.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1.0)
                        delta_q = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1.0)
                        delta_q = 1.0 - (val ** (1.0 / (eta_m + 1.0)))
                    child2[d] += delta_q * (hi[d] - lo[d])
            
            # Clip to bounds
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate children
            fc1 = problem(child1)
            used += 1
            fc2 = problem(child2)
            used += 1
            
            if used >= budget:
                break
            
            # Environmental selection: add children to population, then trim
            # New population
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, fc1, fc2])
            
            # Find non-dominated points
            nondom_idx = non_dominated_sort(F_new)
            
            if len(nondom_idx) <= pop_size:
                # Keep all non-dominated points
                X = X_new[nondom_idx]
                F = F_new[nondom_idx]
                # If we have room, add dominated points
                if len(X) < pop_size:
                    dom_idx = np.setdiff1d(np.arange(len(F_new)), nondom_idx)
                    if len(dom_idx) > 0:
                        # Sort dominated points by sum of objectives
                        sums = np.sum(F_new[dom_idx], axis=1)
                        order = np.argsort(sums)
                        for idx in order:
                            if len(X) >= pop_size:
                                break
                            X = np.vstack([X, X_new[dom_idx[idx]]])
                            F = np.vstack([F, F_new[dom_idx[idx]]])
            else:
                # More non-dominated points than pop_size
                # Trim by hypervolume contribution
                contributions = np.zeros(len(nondom_idx))
                for i, idx in enumerate(nondom_idx):
                    contributions[i] = hv_contribution_2d(F_new[idx], F_new[nondom_idx], ref)
                
                # Keep top pop_size by contribution
                keep = np.argsort(contributions)[-pop_size:]
                X = X_new[nondom_idx[keep]]
                F = F_new[nondom_idx[keep]]
        
        return X

# EVOLVE-BLOCK-END
