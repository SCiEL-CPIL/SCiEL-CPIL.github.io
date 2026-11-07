"""Multi-objective optimizer using differential evolution with HV-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: DE-based search with HV-guided archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with stratified sampling
        n_init = min(30, max(10, budget // 7))
        
        # Stratified sampling: divide each dimension into n_init strata
        X = np.zeros((n_init, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], n_init + 1)
            for i in range(n_init):
                X[i, d] = rng.uniform(edges[i], edges[i + 1])
        
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive
        archive_X = [X[i].copy() for i in range(n_init)]
        archive_F = [F[i].copy() for i in range(n_init)]

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(point, front):
            for f in front:
                if dominates(f, point):
                    return False
            return True

        def update_archive(point, fval, front_X, front_F):
            if not is_non_dominated(fval, front_F):
                return front_X, front_F
            
            dominated_indices = [i for i, f in enumerate(front_F) if dominates(fval, f)]
            new_front_X = [x for i, x in enumerate(front_X) if i not in dominated_indices]
            new_front_F = [f for i, f in enumerate(front_F) if i not in dominated_indices]
            new_front_X.append(point)
            new_front_F.append(fval)
            return new_front_X, new_front_F

        CR = 0.9
        F_start = 0.8
        F_end = 0.3
        pop_size = 20

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            F_mut = F_start + (F_end - F_start) * progress

            if len(archive_F) <= pop_size:
                pop_X = archive_X
                n_pop = len(pop_X)
            else:
                # HV-guided selection: weight by contribution to hypervolume approximation
                norms = np.array([np.sum(f / ref) for f in archive_F])
                weights = 1.0 / (norms + 1e-10)
                weights /= np.sum(weights)
                idxs = rng.choice(len(archive_F), size=pop_size, replace=False, p=weights)
                pop_X = [archive_X[i] for i in idxs]
                n_pop = len(pop_X)

            if n_pop < 3:
                base = rng.choice(n_pop)
                trial = pop_X[base] + F_mut * (hi - lo) * rng.standard_normal(dim)
                trial = np.clip(trial, lo, hi)
            else:
                a = rng.randint(0, n_pop)
                b = rng.randint(0, n_pop)
                while b == a:
                    b = rng.randint(0, n_pop)
                c = rng.randint(0, n_pop)
                while c == a or c == b:
                    c = rng.randint(0, n_pop)
                
                v = pop_X[a] + F_mut * (pop_X[b] - pop_X[c])
                trial = v.copy()
                j_rand = rng.randint(0, dim)
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[a][j]
                trial = np.clip(trial, lo, hi)

            f_trial = problem(trial)
            used += 1

            if is_non_dominated(f_trial, archive_F):
                archive_X, archive_F = update_archive(trial, f_trial, archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END
