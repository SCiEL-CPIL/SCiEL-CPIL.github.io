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
            S = [set() for _ in range(m)]  # Set of individuals dominated by i
            n = [0] * m  # Number of individuals dominating i
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

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return [float('inf')] * m
            
            cd = np.zeros(m)
            for k in range(n_obj):
                obj_vals = np.array([F[idx, k] for idx in indices])
                sorted_indices = np.argsort(obj_vals)
                min_val = obj_vals[sorted_indices[0]]
                max_val = obj_vals[sorted_indices[-1]]
                if max_val == min_val:
                    continue
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (obj_vals[sorted_indices[i + 1]] - obj_vals[sorted_indices[i - 1]]) / (max_val - min_val)
            return cd

        # Helper: Tournament selection
        def tournament_select(X, F, k=2):
            idxs = np.random.choice(len(X), size=k, replace=False)
            # Find non-dominated among selected
            best = idxs[0]
            for idx in idxs[1:]:
                if np.all(F[idx] <= F[best]) and np.any(F[idx] < F[best]):
                    best = idx
            return X[best].copy()

        # Helper: SBX Crossover (Simulated Binary Crossover)
        def sbx_crossover(x1, x2, eta=20.0):
            if np.random.rand() < 0.5:
                return x1.copy(), x2.copy()
            
            x1_new = x1.copy()
            x2_new = x2.copy()
            
            for i in range(dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                    
                if np.random.rand() < 0.5:
                    beta = (2 * np.random.rand()) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * np.random.rand())) ** (1 / (eta + 1))
                    
                x1_new[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                x2_new[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
                # Clip to bounds
                x1_new[i] = np.clip(x1_new[i], lo[i], hi[i])
                x2_new[i] = np.clip(x2_new[i], lo[i], hi[i])
                
            return x1_new, x2_new

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < 0.5:
                    continue
                delta1 = (x_new[i] - lo[i]) / range_[i]
                delta2 = (hi[i] - x_new[i]) / range_[i]
                u = np.random.rand()
                if u < 0.5:
                    xy = (1 + delta1) ** (eta + 1)
                    deltaq = (2 * u + (1 - 2 * u) * xy) ** (1 / (eta + 1)) - 1
                else:
                    xy = (1 + delta2) ** (eta + 1)
                    deltaq = 1 - (2 * (1 - u) + (2 * u - 1) * xy) ** (1 / (eta + 1))
                
                x_new[i] = x_new[i] + deltaq * range_[i]
                x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        while used < budget:
            # Create offspring population
            Q_X = []
            for _ in range(N):
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                Q_X.append(c1)
                Q_X.append(c2)
            
            # We need exactly N offspring for next gen, but we generated 2N.
            # Standard NSGA-II generates N offspring. Let's adjust.
            # Actually, the loop above generates 2N. We should only generate N.
            # Let's restart the offspring generation logic inside the while loop properly.
            
            # Clear Q_X and generate N offspring
            Q_X = []
            for _ in range(N):
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                c1, _ = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                Q_X.append(c1)
            
            # Evaluate offspring
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            if used > budget:
                # This shouldn't happen if we check before, but just in case
                break

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
                    # Partially fill the front
                    remaining = N - len(P_next_X)
                    cd = crowding_distance(R_F, list(front))
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[list(front).index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        P_next_X.append(R_X[idx])
                        P_next_F.append(R_F[idx])
                    break
            
            X = np.array(P_next_X)
            F = np.array(P_next_F)

        return X
