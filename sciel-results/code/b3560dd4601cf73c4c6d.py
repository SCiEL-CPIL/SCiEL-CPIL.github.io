from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer for RE34 using LHS initialization and crowding-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Use a fixed seed for reproducibility of the search path
        # The problem evaluator seeds np.random, but we use a local rng for search
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization: Latin Hypercube Sampling for better initial spread
        pop_size = 30
        if pop_size > budget:
            pop_size = budget
            
        # Generate LHS samples in [0, 1]^dim
        sampler = qmc.LatinHypercube(d=dim, seed=rng)
        lhs_samples = sampler.random(n=pop_size)
        
        # Scale to bounds
        X = lo + lhs_samples * (hi - lo)
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        if used >= budget:
            return X

        # 2. Evolutionary Loop
        while used < budget:
            if time.time() > deadline - 2:
                break
                
            # Identify non-dominated solutions in current population
            nd_idx = self._get_non_dominated_indices(F)
            
            if len(nd_idx) == 0:
                # Fallback: pick random index
                parent_idx = rng.integers(pop_size)
            else:
                # Calculate crowding distance for non-dominated solutions
                # to select a parent from a sparse region
                nd_F = F[nd_idx]
                crowding = self._calc_crowding_distance(nd_F)
                
                # Select parent with highest crowding distance (sparsest region)
                # Add some randomness to avoid deterministic cycles
                # We pick from the top 30% of crowding distances
                threshold = np.percentile(crowding, 70)
                candidates = np.where(crowding >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.arange(len(nd_idx))
                
                local_parent_idx = candidates[rng.integers(len(candidates))]
                parent_idx = nd_idx[local_parent_idx]
            
            parent = X[parent_idx]
            
            # Adaptive Mutation Scale
            # Start with larger steps, shrink over time
            progress = used / budget
            # Cosine annealing from 0.4 to 0.05
            mut_scale = 0.05 + (0.4 - 0.05) * 0.5 * (1 + np.cos(np.pi * progress))
            
            # Gaussian Mutation
            noise = rng.standard_normal(dim)
            child = parent + mut_scale * (hi - lo) * noise
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update Population
            # Strategy: Replace the "worst" individual in the population
            # to maintain diversity.
            # "Worst" is defined by:
            # 1. If dominated, highest sum of objectives.
            # 2. If non-dominated, lowest crowding distance.
            
            # Add child to temp lists for evaluation of replacement
            X_temp = np.vstack([X, child[None, :]])
            F_temp = np.vstack([F, fc[None, :]])
            
            remove_idx = self._find_worst_index_for_removal(F_temp)
            X = np.delete(X_temp, remove_idx, axis=0)
            F = np.delete(F_temp, remove_idx, axis=0)

        # 3. Local Refinement Phase: Polish non-dominated points using L-BFGS-B
        if time.time() < deadline - 10:
            nd_idx = self._get_non_dominated_indices(F)
            if len(nd_idx) > 0:
                nd_F = F[nd_idx]
                crowding = self._calc_crowding_distance(nd_F)
                refine_candidates = nd_idx[np.argsort(crowding)][::-1][:min(5, len(nd_idx))]
                
                for idx in refine_candidates:
                    if time.time() > deadline - 5:
                        break
                    if used >= budget:
                        break
                    
                    parent_x = X[idx].copy()
                    parent_f = F[idx].copy()
                    
                    for _ in range(3):
                        if used >= budget or time.time() > deadline - 3:
                            break
                        
                        w = rng.dirichlet(np.ones(self.n_obj))
                        
                        def scalarized_obj(x, w=w):
                            try:
                                fx = problem(x)
                                return float(np.dot(w, fx))
                            except:
                                return 1e10
                        
                        try:
                            res = minimize(
                                scalarized_obj,
                                parent_x,
                                method='L-BFGS-B',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 50, 'gtol': 1e-6}
                            )
                            
                            if res.fun < np.dot(w, parent_f) - 1e-8:
                                new_x = np.clip(res.x, lo, hi)
                                new_f = np.array(problem(new_x), dtype=np.float64)
                                used += 1
                                
                                X_temp = np.vstack([X, new_x[None, :]])
                                F_temp = np.vstack([F, new_f[None, :]])
                                remove_idx = self._find_worst_index_for_removal(F_temp)
                                X = np.delete(X_temp, remove_idx, axis=0)
                                F = np.delete(F_temp, remove_idx, axis=0)
                        except:
                            pass

        return X

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (vectorized)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        # Vectorized dominance check
        F_i = F[:, None, :]  # (n, 1, m)
        F_j = F[None, :, :]  # (1, n, m)
        
        all_leq = np.all(F_j <= F_i, axis=2)  # (n, n)
        any_lt = np.any(F_j < F_i, axis=2)  # (n, n)
        
        dominates = all_leq & any_lt
        dominated = np.any(dominates, axis=1)
        
        return np.where(~dominated)[0]

    def _calc_crowding_distance(self, F):
        """Calculate crowding distance for a set of non-dominated points F."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        m = F.shape[1]
        crowding = np.zeros(n)
        
        for k in range(m):
            # Sort by objective k
            sorted_indices = np.argsort(F[:, k])
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf
            
            # Normalize objective range
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            f_range = f_max - f_min
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                # Distance between neighbors in sorted order
                dist = F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]
                crowding[sorted_indices[i]] += dist / f_range
                
        return crowding

    def _find_worst_index_for_removal(self, F):
        """
        Given a population F (size N+1), find the index to remove.
        Preference: Remove dominated points first (worst sum).
        If all non-dominated, remove the one with lowest crowding distance.
        """
        n = len(F)
        nd_idx = self._get_non_dominated_indices(F)
        dominated_idx = np.setdiff1d(np.arange(n), nd_idx)
        
        if len(dominated_idx) > 0:
            # Among dominated, remove the one with the worst sum of objectives
            dominated_F = F[dominated_idx]
            sums = np.sum(dominated_F, axis=1)
            worst_local = np.argmax(sums)
            return dominated_idx[worst_local]
        else:
            # All non-dominated: remove the one with lowest crowding distance
            crowding = self._calc_crowding_distance(F)
            return np.argmin(crowding)
# EVOLVE-BLOCK-END
