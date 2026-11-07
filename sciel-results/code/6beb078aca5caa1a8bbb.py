import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        N = 20
        
        # Initialize population
        X = lo + range_ * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            m = len(F)
            S = [set() for _ in range(m)]
            n = [0] * m
            rank = [0] * m
            fronts = []
            front = set()
            
            for i in range(m):
                for j in range(i + 1, m):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n[i] += 1
                
                if n[i] == 0:
                    rank[i] = 0
                    front.add(i)
            
            fronts.append(front)
            k = 0
            while front:
                next_front = set()
                for i in front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            rank[j] = k + 1
                            next_front.add(j)
                k += 1
                front = next_front
                if front:
                    fronts.append(front)
            
            return fronts

        # Generate reference points for NSGA-III
        def generate_reference_points(n_obj, H=10):
            points = []
            for i in range(H + 1):
                for j in range(H - i + 1):
                    k = H - i - j
                    if i + j + k == H:
                        points.append([i/H, j/H, k/H])
            return np.array(points)

        ref_points = generate_reference_points(n_obj)

        # Helper: NSGA-III Selection
        def select_nsga3(F_subset, indices, ref_points, N_sel):
            # F_subset: array of objective values for the partial front
            # indices: list of indices in the combined population corresponding to F_subset rows
            # ref_points: array of reference points
            # N_sel: number of individuals to select
            
            if len(indices) <= N_sel:
                return indices
            
            # Normalize F_subset
            f_min = np.min(F_subset, axis=0)
            f_max = np.max(F_subset, axis=0)
            
            denom = f_max - f_min
            denom[denom == 0] = 1.0
            
            F_norm = (F_subset - f_min) / denom
            
            # Calculate intersection points with hyperplane sum(x) = 1
            intersections = []
            for i in range(len(F_subset)):
                f_prime = F_norm[i]
                sum_f = np.sum(f_prime)
                if sum_f == 0:
                    p = ref_points[0]
                else:
                    p = f_prime / sum_f
                intersections.append(p)
            
            intersections = np.array(intersections)
            
            # Find closest reference point for each intersection
            # dist[i, j] = || intersections[i] - ref_points[j] ||
            dists = np.linalg.norm(intersections[:, np.newaxis, :] - ref_points[np.newaxis, :, :], axis=2)
            
            # For each individual i, min_dist and min_ref_idx
            min_dists = np.min(dists, axis=1)
            min_ref_indices = np.argmin(dists, axis=1)
            
            # Create candidates list: (min_dist, local_idx, ref_idx)
            candidates = []
            for i in range(len(indices)):
                candidates.append((min_dists[i], i, min_ref_indices[i]))
            
            # Sort by distance
            candidates.sort(key=lambda x: x[0])
            
            selected_local = set()
            ref_count = np.zeros(len(ref_points))
            
            # Pass 1: Select one individual per reference point
            for dist, i, r_idx in candidates:
                if ref_count[r_idx] == 0:
                    selected_local.add(i)
                    ref_count[r_idx] += 1
                    if len(selected_local) >= N_sel:
                        break
            
            # Pass 2: Add remaining if needed
            if len(selected_local) < N_sel:
                for dist, i, r_idx in candidates:
                    if i not in selected_local:
                        selected_local.add(i)
                        if len(selected_local) >= N_sel:
                            break
            
            # Map back to original indices
            return [indices[i] for i in selected_local]

        # DE Parameters
        F_mut = 0.8
        CR = 0.5
        
        while used + N <= budget:
            Q_X = []
            for i in range(N):
                # Choose 3 distinct random individuals different from i
                r1, r2, r3 = np.random.choice([j for j in range(N) if j != i], size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                Q_X.append(trial)
            
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(R_F)
            
            # Select next population
            P_next_X = []
            P_next_F = []
            
            for front in fronts:
                if len(P_next_X) + len(front) <= N:
                    for idx in front:
                        P_next_X.append(R_X[idx])
                        P_next_F.append(R_F[idx])
                else:
                    remaining = N - len(P_next_X)
                    # Extract F values for the front
                    F_front = R_F[list(front)]
                    selected_indices = select_nsga3(F_front, list(front), ref_points, remaining)
                    for idx in selected_indices:
                        P_next_X.append(R_X[idx])
                        P_next_F.append(R_F[idx])
                    break
            
            X = np.array(P_next_X)
            F = np.array(P_next_F)
            
            # If we have remaining budget, we can do single evaluations if N > 1
            # But standard DE works in generations. 
            # If budget is not multiple of N, we can just stop or do random perturbations.
            # Given 200 / 20 = 10, it fits perfectly.
            
        return X
