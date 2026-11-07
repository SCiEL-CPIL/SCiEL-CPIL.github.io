"""Multi-objective optimizer using Differential Evolution with Hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Differential Evolution (DE) for multi-objective optimization.
    Uses a hypervolume-based archive to maintain a diverse Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume calculation (from task description)
        ref = np.array([1705.0, 12.0, 0.3])

        # 1. Initialization: Latin Hypercube Sampling (LHS)
        n_init = 30
        X_init = self._lhs(n_init, dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        # Archive: List of (F, X) pairs that are non-dominated
        # We store F and X separately for efficiency
        archive_F = [F_init[i] for i in range(n_init)]
        archive_X = [X_init[i] for i in range(n_init)]
        
        # Clean initial archive
        archive_F, archive_X = self._update_archive(archive_F, archive_X, F_init, X_init)
        
        used = n_init
        all_F = [F_init[i] for i in range(n_init)]
        all_X = [X_init[i] for i in range(n_init)]

        # DE Parameters
        pop_size = 20
        # F (scale factor) starts high for exploration, decays for exploitation
        F_start = 0.8
        F_end = 0.3
        CR = 0.9  # Crossover rate

        # Pre-generate a population for DE (randomly sampled initially, or from archive if small)
        # We will maintain a population of size pop_size
        # Initialize population randomly within bounds
        X_pop = lo + (hi - lo) * np.random.random((pop_size, dim))
        F_pop = np.array([problem(x) for x in X_pop])
        used += pop_size
        
        # Update archive with initial population
        archive_F, archive_X = self._update_archive(archive_F, archive_X, F_pop, X_pop)
        for i in range(pop_size):
            all_F.append(F_pop[i])
            all_X.append(X_pop[i])

        # Main DE Loop
        generation = 0
        while used < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Update F parameter (linear decay)
            progress = used / budget
            F = F_start + (F_end - F_start) * progress

            # Generate offspring for each member of the population
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                # Check budget
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                # DE/rand/1 strategy
                r1, r2, r3 = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Mutation
                v = X_pop[r1] + F * (X_pop[r2] - X_pop[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover
                u = np.zeros(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() <= CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = X_pop[i][j]
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                new_X.append(u)
                new_F.append(f_u)
                
                # Update global lists
                all_F.append(f_u)
                all_X.append(u)

            if not new_X:
                break

            new_X = np.array(new_X)
            new_F = np.array(new_F)

            # Update Population (Greedy Selection: Keep best between parent and child)
            # For multi-objective, we use a simple rule:
            # If child dominates parent -> replace
            # If parent dominates child -> keep
            # If non-dominated -> keep both? 
            # Standard DE for MOO is tricky. 
            # Strategy: We maintain a fixed-size population. 
            # We will select the best 'pop_size' candidates from (Old Pop + New Offspring) 
            # based on hypervolume contribution to the archive or simply non-domination.
            
            # Combine old and new
            X_cand = np.vstack([X_pop, new_X])
            F_cand = np.vstack([F_pop, new_F])
            
            # Select next population
            # We want to keep diversity. 
            # Simple approach: Sort by hypervolume contribution to current archive, or just pick non-dominated.
            # Since pop_size is small (20), we can afford a slightly expensive selection.
            # Let's select the 'pop_size' points from X_cand that are non-dominated with respect to each other,
            # and if there are more, use crowding distance or HV contribution.
            # To keep it fast, let's just pick the 'pop_size' points that have the best "approximate" HV contribution.
            # Approximation: Sum of distances to reference point (lower is better).
            
            # Actually, a more robust simple strategy:
            # 1. Update the global archive with new_F.
            archive_F, archive_X = self._update_archive(archive_F, archive_X, new_F, new_X)
            
            # 2. For the next generation's population, we sample from the archive if it's large enough, 
            #    or use a mix of archive and new non-dominated points.
            #    Let's keep the population as the 'pop_size' best points from the combined set (Old + New)
            #    based on a simple scalarization: min(F) + lambda * diversity?
            #    Simpler: Just keep the 'pop_size' points from the combined set that are non-dominated 
            #    and have the smallest "distance to origin" (proxy for quality).
            
            # Efficient selection for next pop:
            # 1. Identify non-dominated indices in X_cand
            nd_idx = self._non_dominated_indices(F_cand)
            
            if len(nd_idx) >= pop_size:
                # If we have enough non-dominated points, pick the ones with best HV contribution (approx)
                # Or just random from non-dominated to maintain diversity?
                # Let's pick the ones closest to the "ideal" point (min of mins)
                ideal = np.min(F_cand[nd_idx], axis=0)
                dists = np.sum((F_cand[nd_idx] - ideal)**2, axis=1)
                # Sort by distance, take first pop_size
                sorted_idx = nd_idx[np.argsort(dists)][:pop_size]
                X_pop = X_cand[sorted_idx]
                F_pop = F_cand[sorted_idx]
            else:
                # If not enough non-dominated, fill the rest with best dominated points
                # Sort all by distance to ideal
                ideal = np.min(F_cand, axis=0)
                dists_all = np.sum((F_cand - ideal)**2, axis=1)
                sorted_all = np.argsort(dists_all)
                # Ensure we include all non-dominated first?
                # For simplicity, just take the top pop_size from sorted_all
                X_pop = X_cand[sorted_all[:pop_size]]
                F_pop = F_cand[sorted_all[:pop_size]]

            generation += 1

        # Return the best solution found (or any, as return value is ignored, but must not crash)
        # The evaluator records all returned objective vectors from the 'problem' calls.
        # We just need to finish.
        return X_pop

    def _lhs(self, n, dim, lo, hi):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            pts[:, d] = (perm + np.random.random(n)) / n
        return lo + (hi - lo) * pts

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dom[i]:
                continue
            for j in range(n):
                if i == j or is_dom[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
        return np.where(~is_dom)[0]

    def _update_archive(self, archive_F, archive_X, new_F, new_X):
        """
        Update the archive with new points.
        Removes dominated points from the archive.
        Adds new non-dominated points to the archive.
        Returns updated archive_F (list of arrays) and archive_X (list of arrays).
        """
        # Combine
        all_F = np.array(archive_F + list(new_F))
        all_X = np.array(archive_X + list(new_X))
        
        # Find non-dominated indices in the combined set
        nd_idx = self._non_dominated_indices(all_F)
        
        # Extract
        new_archive_F = [all_F[i] for i in nd_idx]
        new_archive_X = [all_X[i] for i in nd_idx]
        
        return new_archive_F, new_archive_X
# EVOLVE-BLOCK-END
