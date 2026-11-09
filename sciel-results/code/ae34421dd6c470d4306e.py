import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        if self.budget < N:
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        V = np.zeros((N, self.dim))
        
        # Initial evaluation
        F = np.array([problem(x) for x in X])
        used = N
        
        # Personal bests
        P_best = X.copy()
        F_best = F.copy()
        
        # Archive
        archive_X = []
        archive_F = []
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def update_archive(f_new, x_new):
            # Check if new point is dominated by any in archive
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            
            # Remove points dominated by new point
            new_archive_F = []
            new_archive_X = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_archive_F.append(archive_F[i])
                    new_archive_X.append(archive_X[i])
            
            new_archive_F.append(f_new)
            new_archive_X.append(x_new)
            
            # Prune if necessary
            if len(new_archive_F) > 20:
                F_arr = np.array(new_archive_F)
                M = len(new_archive_F)
                # Compute minimum distance to other points for each point
                min_dists = np.zeros(M)
                for i in range(M):
                    d = np.linalg.norm(F_arr - F_arr[i], axis=1)
                    d[i] = np.inf # Ignore self
                    min_dists[i] = np.min(d)
                
                # Keep points with largest min distance
                keep_indices = np.argsort(min_dists)[-20:]
                archive_F[:] = [new_archive_F[i] for i in keep_indices]
                archive_X[:] = [new_archive_X[i] for i in keep_indices]
            else:
                archive_F[:] = new_archive_F
                archive_X[:] = new_archive_X

        # Initialize archive
        for i in range(N):
            update_archive(F[i], X[i])
        
        # MOPSO parameters
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5
        c2 = 1.5
        V_max = 0.5 * (hi - lo) # Max velocity is half the range
        
        for g in range(G):
            w = w_max - (w_max - w_min) * (g / max(1, G))
            
            # Select global best from archive
            if len(archive_F) > 0:
                # Choose a leader. To promote diversity, we could pick based on crowding,
                # but random is simpler and often works well.
                # Let's pick a random index.
                g_best_idx = np.random.randint(len(archive_F))
                g_best = archive_X[g_best_idx]
            else:
                g_best = X[0]
            
            for i in range(N):
                # Update velocity
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (g_best - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -V_max, V_max)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Clip to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                used += 1
                
                # Update personal best
                if dominates(F[i], F_best[i]):
                    P_best[i] = X[i].copy()
                    F_best[i] = F[i].copy()
                
                # Update archive
                update_archive(F[i], X[i])
                
        return X
