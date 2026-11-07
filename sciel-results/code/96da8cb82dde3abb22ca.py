import numpy as np
import time

class Optimizer:
            def __init__(self, budget, dim, n_obj):
                self.budget = budget
                self.dim = dim
                self.n_obj = n_obj

            def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 250  # Safety margin
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                span = hi - lo
                
                # Parameters
                N = 20  # Population size
                # Ensure we don't exceed budget in initialization
                if N > self.budget:
                    N = self.budget
                    
                # Initialize
                X = lo + np.random.random((N, self.dim)) * span
                F = np.array([problem(x) for x in X])
                used = N
                
                # Helper: Non-dominated sort
                def non_dominated_sort(F):
                    # Returns list of fronts, each front is a list of indices
                    fronts = []
                    remaining = list(range(len(F)))
                    while remaining:
                        # Find non-dominated in remaining
                        non_dom = []
                        for i in remaining:
                            dominated = False
                            for j in remaining:
                                if i == j: continue
                                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                    dominated = True
                                    break
                            if not dominated:
                                non_dom.append(i)
                        fronts.append(non_dom)
                        remaining = [i for i in remaining if i not in non_dom]
                    return fronts

                # Helper: Crowding distance
                def crowding_distance(F, indices):
                    if len(indices) == 0:
                        return {}
                    dist = {i: 0.0 for i in indices}
                    for m in range(self.n_obj):
                        f_vals = [F[i][m] for i in indices]
                        min_f = min(f_vals)
                        max_f = max(f_vals)
                        if max_f == min_f:
                            for i in indices:
                                dist[i] = float('inf')
                            continue
                        # Sort indices by objective m
                        sorted_indices = sorted(indices, key=lambda i: F[i][m])
                        dist[sorted_indices[0]] = float('inf')
                        dist[sorted_indices[-1]] = float('inf')
                        for k in range(1, len(sorted_indices) - 1):
                            prev = sorted_indices[k-1]
                            next_ = sorted_indices[k+1]
                            dist[sorted_indices[k]] += (F[next_][m] - F[prev][m]) / (max_f - min_f)
                    return dist

                # Main Loop
                while used < self.budget and (time.time() < deadline):
                    # Selection: Tournament
                    # Create next population
                    offspring_X = []
                    offspring_F = []
                    
                    # We generate N offspring
                    for _ in range(N):
                        # Select parents using tournament selection on rank and crowding
                        # First, compute ranks and crowding for current population X, F
                        # Optimization: Compute once per generation
                        fronts = non_dominated_sort(F)
                        # Assign ranks
                        ranks = np.zeros(N, dtype=int)
                        for rank, front in enumerate(fronts):
                            for i in front:
                                ranks[i] = rank
                        # Crowding distance
                        cd = crowding_distance(F, list(range(N)))
                        
                        # Tournament of size 2
                        p1_idx = np.random.randint(N)
                        p2_idx = np.random.randint(N)
                        if ranks[p1_idx] < ranks[p2_idx]:
                            p1 = p1_idx
                        elif ranks[p2_idx] < ranks[p1_idx]:
                            p1 = p2_idx
                        else:
                            p1 = p1_idx if cd[p1_idx] > cd[p2_idx] else p2_idx
                        
                        # Select second parent
                        p3_idx = np.random.randint(N)
                        p4_idx = np.random.randint(N)
                        if ranks[p3_idx] < ranks[p4_idx]:
                            p2 = p3_idx
                        elif ranks[p4_idx] < ranks[p3_idx]:
                            p2 = p4_idx
                        else:
                            p2 = p3_idx if cd[p3_idx] > cd[p4_idx] else p4_idx
                        
                        # Crossover (Blend)
                        x1 = X[p1]
                        x2 = X[p2]
                        alpha = np.random.random()
                        # Linear interpolation with spread
                        child = (1 - alpha) * x1 + alpha * x2
                        # Add mutation
                        # Adaptive mutation: decrease over time
                        progress = used / self.budget
                        sigma = 0.2 * (1 - progress * 0.5) # 0.2 to 0.1
                        child = child + sigma * span * np.random.randn(self.dim)
                        child = np.clip(child, lo, hi)
                        
                        fc = problem(child)
                        used += 1
                        offspring_X.append(child)
                        offspring_F.append(fc)
                        
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break

                    if used >= self.budget or time.time() > deadline:
                        break

                    # Combine parents and offspring
                    X_new = np.vstack([X, np.array(offspring_X)])
                    F_new = np.array(list(F) + offspring_F)
                    
                    # Selection for next generation
                    fronts_new = non_dominated_sort(F_new)
                    next_X = []
                    next_F = []
                    next_indices = []
                    
                    for front in fronts_new:
                        if len(next_X) + len(front) <= N:
                            # Add all
                            for i in front:
                                next_X.append(X_new[i])
                                next_F.append(F_new[i])
                                next_indices.append(i)
                        else:
                            # Need to select from this front based on crowding
                            need = N - len(next_X)
                            cd_front = crowding_distance(F_new, front)
                            # Sort by crowding distance descending
                            sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                            for i in sorted_front[:need]:
                                next_X.append(X_new[i])
                                next_F.append(F_new[i])
                                next_indices.append(i)
                            break
                            
                    X = np.array(next_X)
                    F = np.array(next_F)

                return X
