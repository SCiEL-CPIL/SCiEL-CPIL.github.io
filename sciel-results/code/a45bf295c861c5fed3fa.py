"""Rewritten optimizer for WFG9 (3 objectives, 12 dims) with deadline guard and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Improved evolutionary multi-objective search with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reserve budget for polish
        polish_budget = 40
        evolve_budget = self.budget - polish_budget
        
        # Population size
        pop_size = 30
        max_pop = 50
        
        # Initialization: Random
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of non-dominated points
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def get_non_dominated_indices(F):
            nd_idx = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j: continue
                    if dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx

        # Initialize archive
        nd_idx = get_non_dominated_indices(F)
        archive_X = X[nd_idx].copy()
        archive_F = F[nd_idx].copy()
        
        pop_X = X.copy()
        pop_F = F.copy()
        
        sigma = 0.15
        
        # Evolution Phase
        while used < evolve_budget:
            if time.time() > deadline:
                break
            
            # Decay sigma
            sigma *= 0.99
            
            # Select parent from non-dominated set (archive)
            if len(archive_X) > 0:
                p_idx = rng.integers(len(archive_X))
                parent = archive_X[p_idx]
            else:
                p_idx = rng.integers(len(pop_X))
                parent = pop_X[p_idx]
                
            # Mutate
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update Archive
            is_nd = True
            for af in archive_F:
                if dominates(af, fc):
                    is_nd = False
                    break
            
            if is_nd:
                # Add to archive
                archive_X = np.vstack([archive_X, [child]])
                archive_F = np.vstack([archive_F, [fc]])
                
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for i in range(len(archive_X)):
                    if not dominates(fc, archive_F[i]):
                        new_archive_X.append(archive_X[i])
                        new_archive_F.append(archive_F[i])
                if new_archive_X:
                    archive_X = np.array(new_archive_X)
                    archive_F = np.array(new_archive_F)
                else:
                    archive_X = np.array([child])
                    archive_F = np.array([fc])
                
                # Update population
                pop_X = np.vstack([pop_X, [child]])
                pop_F = np.vstack([pop_F, [fc]])
                
                # Prune population if too large
                if len(pop_X) > max_pop:
                    pop_nd_idx = get_non_dominated_indices(pop_F)
                    if len(pop_nd_idx) >= max_pop:
                        pop_X = pop_X[pop_nd_idx[:max_pop]]
                        pop_F = pop_F[pop_nd_idx[:max_pop]]
                    else:
                        dominated_idx = [i for i in range(len(pop_X)) if i not in pop_nd_idx]
                        if dominated_idx:
                            num_to_add = max_pop - len(pop_nd_idx)
                            chosen = rng.choice(dominated_idx, size=min(num_to_add, len(dominated_idx)), replace=False)
                            keep_idx = np.concatenate([pop_nd_idx, chosen])
                            pop_X = pop_X[keep_idx]
                            pop_F = pop_F[keep_idx]
                        else:
                            pop_X = pop_X[pop_nd_idx]
                            pop_F = pop_F[pop_nd_idx]
            else:
                # Child is dominated. Replace worst in population if better sum
                worst_idx = np.argmax(np.sum(pop_F, axis=1))
                if np.sum(fc) < np.sum(pop_F[worst_idx]):
                    pop_X[worst_idx] = child
                    pop_F[worst_idx] = fc

        # Polish Phase
        # Take top 10 non-dominated points and do local search
        if used < self.budget and len(archive_X) > 0:
            # Sort archive by sum of objectives to pick "best" ones? 
            # Or just pick first 10
            num_polish = min(10, len(archive_X))
            # Pick diverse points? Just pick first num_polish
            polish_points = archive_X[:num_polish]
            
            for p in polish_points:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Local search: try small perturbations
                local_sigma = 0.05
                for _ in range(4): # 4 steps per point
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    candidate = p + local_sigma * (hi - lo) * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    
                    # Check if candidate is non-dominated w.r.t archive
                    is_nd = True
                    for af in archive_F:
                        if dominates(af, fc):
                            is_nd = False
                            break
                    
                    if is_nd:
                        # Add to archive
                        archive_X = np.vstack([archive_X, [candidate]])
                        archive_F = np.vstack([archive_F, [fc]])
                        
                        # Remove dominated
                        new_archive_X = []
                        new_archive_F = []
                        for i in range(len(archive_X)):
                            if not dominates(fc, archive_F[i]):
                                new_archive_X.append(archive_X[i])
                                new_archive_F.append(archive_F[i])
                        if new_archive_X:
                            archive_X = np.array(new_archive_X)
                            archive_F = np.array(new_archive_F)
                        else:
                            archive_X = np.array([candidate])
                            archive_F = np.array([fc])
                        
                        # Update p to be the new candidate for next local step?
                        p = candidate
                        local_sigma *= 0.8

        return archive_X
# EVOLVE-BLOCK-END
