"""Multi-objective optimizer on WFG4 using NSGA-II with Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with Differential Evolution for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a fixed seed for internal RNG to ensure reproducibility of the algorithm's behavior, 
        # though the external seed affects the initial problem state if the problem uses global RNG.
        # The problem function itself is deterministic given x.
        rng = np.random.default_rng(42)

        N = 20  # Population size
        F = 0.8  # DE Mutation Factor
        CR = 0.9  # DE Crossover Rate

        # Initialize Population
        P = lo + (hi - lo) * rng.random((N, self.dim))
        F_obj = np.array([problem(x) for x in P])
        evaluations = N
        
        # Archive of all non-dominated points found so far
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F_list):
            """Check if f_new is dominated by any in archive_F_list."""
            if not archive_F_list:
                return False
            # Vectorized check:
            # f_new is dominated if exists f in archive such that f <= f_new and f != f_new
            # We can do this in a loop for clarity and small sizes
            for f in archive_F_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def remove_dominated(new_f, archive_X_list, archive_F_list):
            """Remove points in archive dominated by new_f. Returns modified lists."""
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F_list)):
                f_i = archive_F_list[i]
                # Check if new_f dominates f_i
                if np.all(new_f <= f_i) and np.any(new_f < f_i):
                    continue # f_i is dominated, skip it
                new_archive_X.append(archive_X_list[i])
                new_archive_F.append(f_i)
            return new_archive_X, new_archive_F

        # Update archive with initial population
        for i in range(N):
            if not is_dominated(F_obj[i], archive_F):
                archive_X, archive_F = remove_dominated(F_obj[i], archive_X, archive_F)
                archive_X.append(P[i])
                archive_F.append(F_obj[i])

        def fast_non_dominated_sort(F_arr):
            """
            Fast non-dominated sort.
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F_arr)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = [0] * n  # n_dom[i] = number of individuals dominating i
            rank = [0] * n   # rank[i] = front index of i
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                
                if n_dom[i] == 0:
                    rank[i] = 0

            fronts = []
            current_front = [i for i in range(n) if rank[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices in F_arr."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        # Main Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Generate Offspring Q using Differential Evolution
            Q = np.zeros_like(P)
            F_Q = np.zeros_like(F_obj)
            
            for i in range(N):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # DE Mutation
                v = P[r1] + F * (P[r2] - P[r3])
                
                # DE Crossover (Binomial)
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        Q[i][j] = v[j]
                    else:
                        Q[i][j] = P[i][j]
                
                # Clip to bounds
                Q[i] = np.clip(Q[i], lo, hi)
                
                # Evaluate
                F_Q[i] = problem(Q[i])
                evaluations += 1
                
                # Check if we exceeded budget mid-generation
                if evaluations >= self.budget:
                    break

            if evaluations >= self.budget:
                break

            # Combine P and Q
            R = np.vstack([P, Q])
            F_R = np.vstack([F_obj, F_Q])
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F_R)
            
            # Select new P
            P_new = []
            F_new = []
            
            for front in fronts:
                if len(P_new) + len(front) <= N:
                    for idx in front:
                        P_new.append(R[idx])
                        F_new.append(F_R[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = N - len(P_new)
                    cd = crowding_distance(F_R, front)
                    # Sort indices by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        P_new.append(R[idx])
                        F_new.append(F_R[idx])
                    break
            
            P = np.array(P_new)
            F_obj = np.array(F_new)
            
            # Update Archive with new population members
            for i in range(N):
                if not is_dominated(F_obj[i], archive_F):
                    archive_X, archive_F = remove_dominated(F_obj[i], archive_X, archive_F)
                    archive_X.append(P[i])
                    archive_F.append(F_obj[i])

        return archive_X
# EVOLVE-BLOCK-END
