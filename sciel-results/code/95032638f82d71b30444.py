import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # Center of the search space
        center = 0.5 * (lo + hi)
        # Radius of the initial hyper-sphere (half the diagonal)
        radius = 0.5 * np.linalg.norm(hi - lo)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_mat, n_total):
            ranks = np.zeros(n_total, dtype=int)
            S = [[] for _ in range(n_total)]
            n_dom = np.zeros(n_total, dtype=int)
            fronts = []

            for i in range(n_total):
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[i], F_mat[j]):
                        S[i].append(j)
                    elif dominates(F_mat[j], F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r

            return ranks, fronts

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Store all evaluated points and their objective values
        all_X = []
        all_F = []
        evals_used = 0

        # Sample points within a shrinking hyper-sphere
        # We will sample in batches to manage memory and time
        # The radius shrinks linearly from `radius` to 0 over the budget
        
        # We'll use a simple loop that generates points and keeps the non-dominated front
        # To be efficient, we'll maintain a list of non-dominated points
        
        # Initialize with a few points
        n_batch = 10
        while evals_used < max_evals:
            if time.time() > deadline:
                break
            
            # Determine current radius based on progress
            progress = evals_used / max_evals
            current_radius = radius * (1.0 - progress)
            
            # Generate batch of points
            X_batch = np.zeros((n_batch, self.dim))
            for i in range(n_batch):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample direction on unit sphere
                u = np.random.randn(self.dim)
                u /= np.linalg.norm(u)
                
                # Sample radius with bias towards smaller radii (cube root distribution)
                # This concentrates samples near the center
                r = current_radius * np.random.rand() ** (1.0 / self.dim)
                
                x = center + r * u
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                evals_used += 1
                
            if evals_used >= max_evals:
                break
                
            # Periodically prune the archive to keep it manageable
            # We'll keep the non-dominated front
            if len(all_F) > 100:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                
                ranks, fronts = non_dominated_sort(F_arr, len(all_F))
                
                # Select top 50 individuals based on rank and crowding distance
                selected_indices = []
                for front in fronts:
                    if len(selected_indices) + len(front) <= 50:
                        selected_indices.extend(front)
                    else:
                        remaining = 50 - len(selected_indices)
                        dists = crowding_distance(F_arr, front)
                        front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                        selected_indices.extend(front_sorted[:remaining])
                        break
                        
                all_X = [X_arr[i] for i in selected_indices]
                all_F = [F_arr[i] for i in selected_indices]

        return np.array(all_X) if all_X else np.zeros((0, self.dim))
