"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(12345)
        
        # Reference point for scoring
        ref = np.array([3.0, 5.0, 7.0])
        
        # Archive to store non-dominated points
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is better than ref
            if not np.all(f < ref):
                return False
            
            # Check if f is dominated by any point in archive
            dominated = False
            to_remove = []
            for i, af in enumerate(archive_F):
                if dominates(af, f):
                    dominated = True
                    break
                if dominates(f, af):
                    to_remove.append(i)
            
            if dominated:
                return False
            
            # Remove dominated points from archive
            for i in reversed(to_remove):
                archive_X.pop(i)
                archive_F.pop(i)
                
            archive_X.append(x)
            archive_F.append(f)
            return True

        # Initial Population
        pop_size = 20
        remaining = self.budget
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        remaining -= pop_size
        
        for i in range(pop_size):
            add_to_archive(X[i], F[i])
            
        # Helper: Compute crowding distance for archive
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
            return cd

        # Main Loop
        while remaining > 0:
            if time.time() > deadline:
                break
            
            # If archive is empty, restart with random points
            if len(archive_F) == 0:
                X = lo + (hi - lo) * rng.random((10, self.dim))
                F = np.array([problem(x) for x in X])
                remaining -= 10
                for i in range(10):
                    add_to_archive(X[i], F[i])
                continue
            
            # Selection: Pick 2 parents from archive using tournament selection on hypervolume contribution proxy
            # Proxy: Lower sum of objectives + high crowding distance
            cd = crowding_distance(np.array(archive_F))
            
            # Normalize objectives for scoring
            F_arr = np.array(archive_F)
            min_F = F_arr.min(axis=0)
            max_F = F_arr.max(axis=0)
            range_F = max_F - min_F
            range_F[range_F == 0] = 1.0
            norm_F = (F_arr - min_F) / range_F
            obj_score = norm_F.sum(axis=1)
            
            # Combine objective score and crowding distance
            # Higher crowding is better, lower obj_score is better
            # We want to select diverse and good points
            # Let's use a weighted sum: score = obj_score - alpha * cd
            # But cd can be inf. Cap it.
            cd_capped = np.minimum(cd, 10.0)
            # Normalize cd to [0,1] roughly
            max_cd = cd_capped.max()
            if max_cd > 0:
                cd_norm = cd_capped / max_cd
            else:
                cd_norm = cd_capped
            
            selection_score = obj_score - 0.5 * cd_norm
            
            # Tournament selection
            def select_parent():
                idx1 = rng.integers(len(archive_F))
                idx2 = rng.integers(len(archive_F))
                if selection_score[idx1] < selection_score[idx2]:
                    return idx1
                else:
                    return idx2
            
            p1_idx = select_parent()
            p2_idx = select_parent()
            
            x1 = np.array(archive_X[p1_idx])
            x2 = np.array(archive_X[p2_idx])
            
            # Crossover: SBX or Simple Blend
            # Use simple blend crossover
            beta = rng.random()
            child1 = beta * x1 + (1 - beta) * x2
            child2 = (1 - beta) * x1 + beta * x2
            
            # Mutation: Adaptive Gaussian
            # Scale sigma based on current archive spread
            if len(archive_F) > 1:
                spread = (np.array(archive_F).max(axis=0) - np.array(archive_F).min(axis=0))
            else:
                spread = np.ones(self.n_obj) * 1.0
            
            # Estimate step size based on bounds and remaining budget
            # Start large, shrink as budget depletes? Or keep constant?
            # WFG4 is multi-modal. Let's use a moderate step size.
            sigma = 0.1 * (hi - lo)
            
            child1 = child1 + sigma * rng.standard_normal(self.dim)
            child2 = child2 + sigma * rng.standard_normal(self.dim)
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            f1 = problem(child1)
            remaining -= 1
            if remaining <= 0:
                break
            f2 = problem(child2)
            remaining -= 1
            
            add_to_archive(child1, f1)
            add_to_archive(child2, f2)
            
        return archive_X

# EVOLVE-BLOCK-END
