import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize particles
        X = np.random.uniform(lo, hi, size=(N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Personal bests
        X_pbest = X.copy()
        F_pbest = F.copy()
        
        # Velocity initialization
        v_max = 0.2 * range_
        v_min = -0.2 * range_
        V = np.random.uniform(v_min, v_max, size=(N, self.dim))
        
        # Global Archive
        archive_X = []
        archive_F = []
        
        def _is_dominated(a, b):
            return np.all(a <= b) and np.any(a < b)
            
        def _update_archive(archive_X, archive_F, new_X, new_F, max_size=40):
            # Combine
            all_X = np.vstack([archive_X, new_X]) if len(archive_X) > 0 else new_X
            all_F = np.vstack([archive_F, new_F]) if len(archive_F) > 0 else new_F
            
            # Filter non-dominated
            non_dom_indices = []
            for i in range(len(all_F)):
                dominated = False
                for j in range(len(all_F)):
                    if i != j and _is_dominated(all_F[i], all_F[j]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_indices.append(i)
                    
            if not non_dom_indices:
                return archive_X, archive_F
                
            non_dom_X = all_X[non_dom_indices]
            non_dom_F = all_F[non_dom_indices]
            
            # If size exceeds max, select most diverse
            if len(non_dom_X) > max_size:
                # Calculate crowding distance
                crowd = np.zeros(len(non_dom_F))
                for k in range(self.n_obj):
                    sorted_idx = np.argsort(non_dom_F[:, k])
                    crowd[sorted_idx[0]] = np.inf
                    crowd[sorted_idx[-1]] = np.inf
                    min_val = non_dom_F[sorted_idx[0], k]
                    max_val = non_dom_F[sorted_idx[-1], k]
                    range_val = max_val - min_val
                    if range_val == 0: continue
                    for j in range(1, len(non_dom_F)-1):
                        idx = sorted_idx[j]
                        idx_prev = sorted_idx[j-1]
                        idx_next = sorted_idx[j+1]
                        crowd[sorted_idx[j]] += (non_dom_F[idx_next, k] - non_dom_F[idx_prev, k]) / range_val
                        
                # Select top max_size by crowding distance
                top_indices = np.argsort(crowd)[::-1][:max_size]
                non_dom_X = non_dom_X[top_indices]
                non_dom_F = non_dom_F[top_indices]
                
            return non_dom_X, non_dom_F

        # Initial archive update
        archive_X, archive_F = _update_archive([], [], X, F)
        
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        while used + N <= self.budget:
            # Update velocities and positions
            r1 = np.random.rand(N, 1)
            r2 = np.random.rand(N, 1)
            
            # Select global leaders from archive
            # To encourage diversity, we can pick random leaders or leaders with high crowding
            # Let's pick random leaders for simplicity and robustness in small budgets
            leader_indices = np.random.randint(0, len(archive_X), size=N)
            X_gbest = archive_X[leader_indices]
            
            V = w * V + c1 * r1 * (X_pbest - X) + c2 * r2 * (X_gbest - X)
            
            # Clip velocities
            V = np.clip(V, v_min, v_max)
            
            X_new = X + V
            X_new = np.clip(X_new, lo, hi)
            
            F_new = np.array([problem(x) for x in X_new])
            used += N
            
            # Update personal bests
            for i in range(N):
                if _is_dominated(F_new[i], F_pbest[i]):
                    X_pbest[i] = X_new[i]
                    F_pbest[i] = F_new[i]
                # Note: We do not replace if they are non-dominated to each other in simple MOPSO,
                # but standard MOPSO often just keeps the first non-dominated or uses a strategy.
                # For simplicity and to ensure progress, if they are non-dominated, we keep the old one
                # unless we implement a more complex pbest strategy. 
                # Actually, a better strategy: if non-dominated, keep both? No, pbest is single.
                # Let's stick to: update if dominated. If non-dominated, keep old (conservative) or random.
                # To avoid stagnation, let's add a small probability of replacing even if non-dominated?
                # No, let's keep it simple.
            
            # Update Archive
            # We add the new non-dominated solutions from F_new to the archive
            # But we should only add solutions that are not dominated by the archive or other new ones.
            # _update_archive handles this by combining and filtering.
            # However, passing all X_new might be inefficient if N is large, but N=20 is fine.
            # But we should only add the ones that are potentially non-dominated.
            # Let's just pass the new ones.
            archive_X, archive_F = _update_archive(archive_X, archive_F, X_new, F_new)
            
            # Update X to X_new for next iteration
            X = X_new
            F = F_new
            
        return X
