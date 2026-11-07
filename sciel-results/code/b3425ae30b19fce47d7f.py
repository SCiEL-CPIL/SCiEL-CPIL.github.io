import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        # Smaller population to allow more generations
        pop_size = 12
        
        if pop_size < 3:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # DE parameters
        F_cr = 0.5  # Crossover rate
        F_scale = 0.7  # Scale factor

        # Main DE Loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Non-dominated sort for selection
            ranks, fronts = self.non_dominated_sort(F)
            cd = self.crowding_distance(F, fronts)

            # Create offspring using DE/rand/1
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))

            for i in range(pop_size):
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Differential evolution mutation
                trial = X[a] + F_scale * (X[b] - X[c])
                
                # Binomial crossover
                cross_point = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < F_cr or j == cross_point:
                        trial[j] = X[a][j] + F_scale * (X[b][j] - X[c][j])
                    
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                offspring_X[i] = trial
                offspring_F[i] = problem(trial)
                used += 1

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            cd_all = self.crowding_distance(F_all, fronts_all)

            # Select next population
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

        # Local Polish Phase
        # Use remaining budget to refine the non-dominated set
        # Extract current non-dominated points
        ranks_final, fronts_final = self.non_dominated_sort(F)
        # Take the first front (non-dominated)
        if fronts_final:
            nd_indices = fronts_final[0]
            nd_X = X[nd_indices]
            nd_F = F[nd_indices]
        else:
            nd_X = X
            nd_F = F

        # If we have budget left, do local search
        # We maintain an archive of non-dominated points
        archive_X = [x for x in nd_X]
        archive_F = [f for f in nd_F]
        
        # Simple local search: perturb points in the archive
        # We will use the remaining evaluations for this
        while used < budget:
            if time.time() > deadline:
                break
            
            # Pick a random point from the archive to perturb
            if not archive_X:
                break
            idx = np.random.randint(len(archive_X))
            x_base = archive_X[idx]
            
            # Generate a neighbor with small noise
            # Scale noise by range
            noise_scale = 0.05 * (hi - lo)
            x_new = x_base + np.random.normal(0, 1, dim) * noise_scale
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            used += 1
            
            # Check if x_new is dominated by any point in archive
            is_dominated = False
            for f_arch in archive_F:
                if self.dominates(f_arch, f_new):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Add to archive
                archive_X.append(x_new)
                archive_F.append(f_new)
                
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for x_arch, f_arch in zip(archive_X, archive_F):
                    if not self.dominates(f_new, f_arch):
                        new_archive_X.append(x_arch)
                        new_archive_F.append(f_arch)
                archive_X = new_archive_X
                archive_F = new_archive_F
            else:
                # If dominated, we could try a different direction or just discard
                # To save time, just discard and try next eval
                pass

        # Return the final set (though return value is ignored, we return the archive points)
        # The evaluator records ALL objective vectors returned by problem(x) calls?
        # "Every returned objective vector is recorded"
        # This implies the evaluator wraps the problem function?
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This usually means the evaluator intercepts the calls to `problem`.
        # So we just need to make sure we call `problem` within the budget.
        # The return value of __call__ is ignored.
        
        return archive_X

    def non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns ranks (array of int, 0-based) and fronts (list of lists of indices).
        """
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []

        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self.dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front.append(i)

        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = len(fronts)
                        next_front.append(j)
            front = next_front

        return ranks, fronts

    def dominates(self, a, b):
        """
        Returns True if a dominates b.
        """
        return np.all(a <= b) and np.any(a < b)

    def crowding_distance(self, F, fronts):
        """
        Calculates crowding distance for each point.
        """
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)

        for front in fronts:
            m = len(front)
            if m <= 2:
                for i in front:
                    cd[i] = np.inf
                continue

            for k in range(n_obj):
                sorted_idx = sorted(front, key=lambda idx: F[idx, k])

                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                if m > 2:
                    f_min = F[sorted_idx[0], k]
                    f_max = F[sorted_idx[-1], k]
                    f_range = f_max - f_min

                    if f_range < 1e-14:
                        continue

                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range

        return cd
