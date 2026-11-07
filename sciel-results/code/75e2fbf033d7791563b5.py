import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        # Determine population size and generations to fit budget
        # Total evals = pop_size * (n_gen + 1)
        # Let's aim for pop_size = 20
        pop_size = 20
        max_n_gen = (self.budget // pop_size) - 1
        if max_n_gen < 1:
            pop_size = self.budget // 2
            max_n_gen = 1
            
        n_gen = max_n_gen
        
        # Initialize
        X = lo + np.random.rand(pop_size, self.dim) * range_vec
        V = np.zeros((pop_size, self.dim))
        
        # Helper: Check dominance
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper: Crowding Distance
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                f_vals = F[:, m]
                min_val = np.min(f_vals)
                max_val = np.max(f_vals)
                if max_val == min_val:
                    continue
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i-1]
                    next_idx = sorted_idx[i+1]
                    dist[idx] += (f_vals[next_idx] - f_vals[prev_idx]) / (max_val - min_val)
            return dist

        # Initialize evaluations
        F = np.array([problem(x) for x in X])
        
        # Initialize pbest
        pbest_F = F.copy()
        pbest_X = X.copy()
        
        # Initialize Archive
        archive_F = []
        archive_X = []
        for i in range(pop_size):
            if any(dominates(af, F[i]) for af in archive_F):
                continue
            new_af = []
            new_ax = []
            for j in range(len(archive_F)):
                if not dominates(F[i], archive_F[j]):
                    new_af.append(archive_F[j])
                    new_ax.append(archive_X[j])
            new_af.append(F[i])
            new_ax.append(X[i])
            archive_F = new_af
            archive_X = new_ax
            
        if not archive_F:
            archive_F = [F[i] for i in range(pop_size)]
            archive_X = [X[i] for i in range(pop_size)]

        # MOPSO Parameters
        w_min = 0.4
        w_max = 0.9
        c1 = 2.0
        c2 = 2.0
        max_archive_size = 50
        
        for g in range(n_gen):
            # Update w
            w = w_min + (w_max - w_min) * (1 - g / n_gen)
            
            # 1. Update Personal Bests
            for i in range(pop_size):
                if dominates(F[i], pbest_F[i]):
                    pbest_F[i] = F[i]
                    pbest_X[i] = X[i]

            # 2. Update Global Archive
            all_F = np.array(archive_F + [F[i] for i in range(pop_size)])
            all_X = np.array(archive_X + [X[i] for i in range(pop_size)])
            
            non_dom_indices = []
            n_all = len(all_F)
            for i in range(n_all):
                is_dom = False
                for j in range(n_all):
                    if i == j:
                        continue
                    if dominates(all_F[j], all_F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_indices.append(i)
            
            archive_F = [all_F[i] for i in non_dom_indices]
            archive_X = [all_X[i] for i in non_dom_indices]
            
            if len(archive_F) > max_archive_size:
                F_arr = np.array(archive_F)
                cd = crowding_distance(F_arr)
                sorted_idx = np.argsort(-cd)
                selected_idx = sorted_idx[:max_archive_size]
                archive_F = [F_arr[i] for i in selected_idx]
                archive_X = [archive_X[i] for i in selected_idx]

            # 3. Update Velocity and Position
            new_X = np.zeros_like(X)
            for i in range(pop_size):
                if len(archive_F) > 0:
                    F_arr = np.array(archive_F)
                    cd = crowding_distance(F_arr)
                    if len(archive_F) == 1:
                        g_best_idx = 0
                    else:
                        idx1 = np.random.randint(len(archive_F))
                        idx2 = np.random.randint(len(archive_F))
                        if cd[idx1] > cd[idx2]:
                            g_best_idx = idx1
                        else:
                            g_best_idx = idx2
                    g_best_X = archive_X[g_best_idx]
                else:
                    g_best_X = X[i]

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (g_best_X - X[i])
                new_X[i] = X[i] + V[i]
                new_X[i] = np.clip(new_X[i], lo, hi)

            # 4. Evaluate New Population
            X = new_X
            F = np.array([problem(x) for x in X])

        return X
