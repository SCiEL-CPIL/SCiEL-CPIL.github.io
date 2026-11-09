"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-style evolutionary search with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        rng = np.random.default_rng(0)

        # Structured initialization: corners, midpoints, random
        init_points = []
        
        # All 32 corners
        for i in range(32):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)
        
        # Midpoints
        init_points.append((lo + hi) / 2)
        
        # Add random points to fill up to population size
        pop_size = 32
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))
        
        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        def is_dominated_by(f, ref_F):
            """Check if f is dominated by any in ref_F."""
            for af in ref_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, g):
            """Check if f dominates g."""
            return np.all(f <= g) and np.any(f < g)

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sorting. Returns list of fronts."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # Solutions dominated by i
            nd = [0] * n  # Number of solutions dominating i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F_arr[i], F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif dominates(F_arr[j], F_arr[i]):
                        S[j].append(i)
                        nd[i] += 1
            
            current_front = [i for i in range(n) if nd[i] == 0]
            if not current_front:
                return []
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def crowding_distance(front, F_arr):
            """Calculate crowding distance for a front."""
            if not front:
                return np.zeros(len(F_arr))
            
            n = len(front)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = sorted(front, key=lambda i: F_arr[i, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F_arr[sorted_idx[0], m]
                f_max = F_arr[sorted_idx[-1], m]
                if f_max - f_min > 1e-10:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (F_arr[sorted_idx[k + 1], m] - F_arr[sorted_idx[k - 1], m]) / (f_max - f_min)
            
            return cd

        # Initial archive
        archive_X = []
        archive_F = []
        for i in range(pop_size):
            if not is_dominated_by(F[i], [f for f in archive_F]):
                # Remove dominated from archive
                new_ax = []
                new_af = []
                for ax, af in zip(archive_X, archive_F):
                    if not dominates(F[i], af):
                        new_ax.append(ax)
                        new_af.append(af)
                archive_X = new_ax
                archive_F = new_af
                archive_X.append(X[i].copy())
                archive_F.append(F[i].copy())

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() - start_time > deadline:
                break

            # Generate offspring population
            offspring_X = []
            offspring_F = []

            # Binary tournament selection
            def tournament_select(pop_X, pop_F, pop_size):
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                fi = pop_F[i]
                fj = pop_F[j]
                if dominates(fi, fj):
                    return i
                elif dominates(fj, fi):
                    return j
                else:
                    # Use sum as tiebreaker
                    return i if np.sum(fi) <= np.sum(fj) else j

            # Create offspring
            for _ in range(pop_size):
                # Select parent using tournament
                parent_idx = tournament_select(X, F, pop_size)
                parent = X[parent_idx]

                # DE/CR mutation
                idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
                while idx1 == idx2 or idx2 == idx3 or idx1 == idx3:
                    idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
                
                F_factor = 0.5 + 0.5 * rng.random()
                CR = 0.5 + 0.5 * rng.random()
                
                mutant = X[idx1] + F_factor * (X[idx2] - X[idx3])
                
                # Binomial crossover
                r = rng.random(dim)
                j_rand = rng.integers(dim)
                trial = np.where((r < CR) | (np.arange(dim) == j_rand), mutant, parent)
                trial = np.clip(trial, lo, hi)
                
                ft = problem(trial)
                used += 1
                
                offspring_X.append(trial)
                offspring_F.append(ft)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = 2 * pop_size

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            next_fronts = []
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    next_fronts.append(front)
                else:
                    # Need to truncate using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(front, combined_F)
                    sorted_by_cd = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_by_cd[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)

            # Update archive
            for i in range(pop_size):
                if not is_dominated_by(F[i], [f for f in archive_F]):
                    new_ax = []
                    new_af = []
                    for ax, af in zip(archive_X, archive_F):
                        if not dominates(F[i], af):
                            new_ax.append(ax)
                            new_af.append(af)
                    archive_X = new_ax
                    archive_F = new_af
                    archive_X.append(X[i].copy())
                    archive_F.append(F[i].copy())

        return X
# EVOLVE-BLOCK-END
