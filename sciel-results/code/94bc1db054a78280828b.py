import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Strategy: Grid-based decomposition with Latin Hypercube Sampling (LHS)
        # to ensure maximal spatial coverage of the Pareto front manifold.
        
        def eval_point(x):
            return np.array(problem(x))
        
        # Phase 1: Initial Coverage using LHS
        # We split the budget into an initial exploration phase and a refinement phase.
        # LHS ensures that each dimension is sampled uniformly across its range,
        # which is excellent for discovering the general shape of the Pareto front.
        
        # Let's use 50% of the budget for initial LHS exploration.
        n_lhs = self.budget // 2
        if n_lhs < 10:
            n_lhs = 10
        if n_lhs > self.budget:
            n_lhs = self.budget
            
        # Generate LHS points
        # For each dimension, we divide the range into n_lhs bins and sample one point from each bin.
        # Then we shuffle the bins for each dimension to create the points.
        
        X_lhs = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Create bins
            bin_edges = np.linspace(lo[i], hi[i], n_lhs + 1)
            # Sample one point from each bin
            samples = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) * np.random.rand(n_lhs)
            # Shuffle
            np.random.shuffle(samples)
            X_lhs[:, i] = samples
            
        # Evaluate LHS points
        F_lhs = np.array([eval_point(x) for x in X_lhs])
        
        # Check time
        if time.time() > deadline - 5.0:
            return X_lhs
            
        # Phase 2: Local Refinement
        # Use the remaining budget to refine around the best points found in Phase 1.
        # We identify the non-dominated points from Phase 1 and perform local searches around them.
        
        remaining_budget = self.budget - n_lhs
        if remaining_budget <= 0:
            return X_lhs
            
        # Find non-dominated points in F_lhs
        fronts = self._fast_non_dominated_sort(F_lhs)
        
        # Select the best front (fronts[0]) for refinement
        if fronts:
            best_front_indices = list(fronts[0])
        else:
            best_front_indices = list(range(n_lhs))
            
        # Limit the number of centers for refinement to avoid too many small steps
        # We want to use the remaining budget effectively.
        # Let's say we pick up to 10 centers.
        max_centers = min(10, len(best_front_indices))
        if max_centers > 0:
            # Randomly select from the best front
            selected_centers = np.random.choice(best_front_indices, max_centers, replace=False)
            
            # Distribute remaining budget among centers
            # If we have R budget and C centers, we do R/C evaluations per center.
            evals_per_center = remaining_budget // max_centers
            if evals_per_center < 1:
                evals_per_center = 1
                
            # Adjust for any leftover
            total_refine = evals_per_center * max_centers
            if total_refine > remaining_budget:
                evals_per_center = (remaining_budget // max_centers)
                if evals_per_center < 1:
                    evals_per_center = 1
                total_refine = evals_per_center * max_centers
                
            # Perform local search (e.g., random perturbations or Nelder-Mead)
            # Given the short time, simple random perturbations with decreasing step size might be safer.
            # Or a small population-based local search.
            
            # Let's use a simple local search: for each center, generate evals_per_center random points in a small neighborhood.
            # The neighborhood size should be small, e.g., 5% of the range.
            step_size = 0.05 * (hi - lo)
            
            X_refine = []
            F_refine = []
            
            for center_idx in selected_centers:
                center = X_lhs[center_idx]
                for _ in range(evals_per_center):
                    if time.time() > deadline - 1.0:
                        break
                    # Generate a random point in the neighborhood
                    # Uniform distribution in a hypercube around the center
                    offsets = (np.random.rand(dim) - 0.5) * 2 * step_size
                    x_new = center + offsets
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = eval_point(x_new)
                    X_refine.append(x_new)
                    F_refine.append(f_new)
                    
            if X_refine:
                X_refine = np.array(X_refine)
                F_refine = np.array(F_refine)
            else:
                X_refine = np.empty((0, dim))
                F_refine = np.empty((0, n_obj))
        else:
            X_refine = np.empty((0, dim))
            F_refine = np.empty((0, n_obj))
            
        # Combine results
        if len(X_refine) > 0:
            X_all = np.vstack([X_lhs, X_refine])
            F_all = np.vstack([F_lhs, F_refine])
        else:
            X_all = X_lhs
            F_all = F_lhs
            
        return X_all

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
            
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                    
        front = set()
        for i in range(n):
            if n_dom[i] == 0:
                front.add(i)
        if front:
            fronts.append(front)
                
        k = 0
        while k < len(fronts):
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts
