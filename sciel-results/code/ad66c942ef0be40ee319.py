import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290 # 10s margin
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            range_ = hi - lo
            
            # LHS Initialization
            pop_size = 20
            # Use global random state for reproducibility if needed, but default_rng is fine
            rng = np.random.default_rng(0)
            
            # LHS
            X = np.zeros((pop_size, self.dim))
            for i in range(self.dim):
                intervals = np.linspace(0, 1, pop_size + 1)
                points = rng.uniform(intervals[:-1], intervals[1:])
                rng.shuffle(points)
                X[:, i] = points
            X = lo + range_ * X
            
            F = np.array([problem(x) for x in X])
            used = pop_size
            
            def get_nondominated(F_mat):
                n = len(F_mat)
                nondom = np.ones(n, dtype=bool)
                for i in range(n):
                    if not nondom[i]: continue
                    for j in range(n):
                        if i == j or not nondom[j]: continue
                        if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                            nondom[i] = False
                            break
                return np.where(nondom)[0]
            
            def crowding_distance(F_mat, indices):
                if len(indices) <= 2:
                    return np.full(len(indices), np.inf)
                dist = np.zeros(len(indices))
                F_sub = F_mat[indices]
                n_obj = F_sub.shape[1]
                for k in range(n_obj):
                    f_k = F_sub[:, k]
                    min_f, max_f = np.min(f_k), np.max(f_k)
                    if max_f == min_f: continue
                    sorted_idx = np.argsort(f_k)
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        idx_curr = sorted_idx[i]
                        idx_prev = sorted_idx[i-1]
                        idx_next = sorted_idx[i+1]
                        dist[idx_curr] += (f_k[idx_next] - f_k[idx_prev]) / (max_f - min_f)
                return dist

            while used < self.budget:
                if time.time() > deadline:
                    break
                
                nondom_idx = get_nondominated(F)
                
                # Select parent
                if len(nondom_idx) > 0:
                    parent_idx = nondom_idx[rng.integers(len(nondom_idx))]
                else:
                    parent_idx = rng.integers(len(F))
                    
                parent = X[parent_idx]
                
                # Mutation
                sigma = 0.1
                child = parent + sigma * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replacement
                # 1. Does child dominate any in pop?
                dominated_by_child = []
                for i in range(len(F)):
                    if np.all(fc <= F[i]) and np.any(fc < F[i]):
                        dominated_by_child.append(i)
                
                if dominated_by_child:
                    # Replace the one with worst crowding distance among dominated?
                    # Or just the first one?
                    # To maintain diversity, we should replace the one that is "worst" in terms of crowding.
                    # But they are all dominated, so they are "bad".
                    # Let's just replace the one with the largest sum of objectives (simplest).
                    # Or better: replace the one that is dominated by child AND has the largest crowding distance in the *global* non-dominated set?
                    # No, if it's dominated, it's not in the non-dominated set.
                    # Let's replace the dominated point with the largest crowding distance *if we calculate it*.
                    # Actually, standard NSGA-II replaces the dominated point. If multiple, pick one with largest crowding distance in the *current* front?
                    # They are not in the front.
                    # Let's just pick the one with the largest sum.
                    worst_dom = max(dominated_by_child, key=lambda i: np.sum(F[i]))
                    X[worst_dom] = child
                    F[worst_dom] = fc
                else:
                    # Check if child is non-dominated
                    is_nondom = True
                    for i in range(len(F)):
                        if np.all(F[i] <= fc) and np.any(F[i] < fc):
                            is_nondom = False
                            break
                    
                    if is_nondom:
                        # Child is non-dominated.
                        # Add to front.
                        # If front size > pop_size, we need to remove one.
                        # But we are doing 1-for-1 replacement.
                        # So we must remove one existing point.
                        # Which one? The one with the largest crowding distance in the *new* non-dominated set.
                        # New set = old non-dominated set + child.
                        # But we don't have the new set in X/F yet.
                        # We can simulate:
                        # 1. Get old nondom indices.
                        # 2. Add child to a temporary list.
                        # 3. Calculate crowding on this temporary list.
                        # 4. Remove the one with max crowding.
                        
                        # This is slightly expensive but pop_size is 20.
                        
                        # However, if the child is non-dominated, it *expands* the Pareto front.
                        # We should replace the point in the *current* non-dominated front that has the largest crowding distance.
                        # This is the standard NSGA-II approach when the front is full.
                        
                        if len(nondom_idx) > 0:
                            # Calculate crowding on current nondom set
                            dists = crowding_distance(F, nondom_idx)
                            worst_front_idx = nondom_idx[np.argmax(dists)]
                            X[worst_front_idx] = child
                            F[worst_front_idx] = fc
                        else:
                            # Should not happen if is_nondom is true and pop exists
                            pass
                    else:
                        # Child is dominated, discard
                        pass
                        
            return X
