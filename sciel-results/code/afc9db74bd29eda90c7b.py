import numpy as np
import time

class Optimizer:
    """NSGA-II Multi-objective Optimizer"""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: ~10% of budget, min 10, max 20
        pop_size = max(10, min(20, self.budget // 10))
        
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        # eta_m: distribution index for mutation (higher = less aggressive)
        # eta_c: distribution index for crossover (higher = more aggressive)
        eta_m = 20.0
        eta_c = 20.0
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sorting
            ranks = self._non_dominated_sort(F)
            
            # 2. Crowding distance
            crowd_dist = self._crowding_distance(F, ranks)
            
            # 3. Selection: Tournament selection for parents
            # We need to generate pop_size children. 
            # Each child comes from 2 parents.
            # So we need 2 * pop_size parent selections.
            parents = []
            for _ in range(2 * pop_size):
                # Tournament size 2
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if self._tournament_select(ranks, crowd_dist, i1, i2):
                    parents.append(X[i1])
                else:
                    parents.append(X[i2])
            
            # 4. Crossover and Mutation
            children_X = []
            for i in range(0, 2 * pop_size, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, eta_m)
                c2 = self._polynomial_mutation(c2, lo, hi, eta_m)
                
                children_X.append(c1)
                children_X.append(c2)
            
            children_X = np.array(children_X)
            
            # Evaluate children
            children_F = np.array([problem(x) for x in children_X])
            used += pop_size
            
            # 5. Combine parents and children
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            
            # 6. Selection for next generation
            combined_ranks = self._non_dominated_sort(combined_F)
            combined_crowd = self._crowding_distance(combined_F, combined_ranks)
            
            next_X = []
            next_F = []
            selected_indices = np.zeros(len(combined_F), dtype=bool)
            
            # Fill next generation by rank
            for rank in range(max(combined_ranks) + 1):
                rank_indices = np.where(combined_ranks == rank)[0]
                if len(next_X) + len(rank_indices) <= pop_size:
                    # Take all
                    for idx in rank_indices:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    selected_indices[rank_indices] = True
                else:
                    # Take only some based on crowding distance
                    needed = pop_size - len(next_X)
                    # Sort by crowding distance (descending)
                    crowd_vals = combined_crowd[rank_indices]
                    sorted_indices = rank_indices[np.argsort(crowd_vals)[::-1]]
                    for idx in sorted_indices[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    selected_indices[rank_indices] = False # Mark rest as not selected, though we stop anyway
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [set() for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of individuals dominating i
        front = [set()]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        front_num = 0
        while True:
            current_front = set()
            for i in range(n):
                if n_dom[i] == 0 and ranks[i] == 0 and i not in front[front_num]:
                    current_front.add(i)
            
            if not current_front:
                break
                
            for i in current_front:
                ranks[i] = front_num
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        # j will be in the next front
                        pass # We don't add to current_front, we add to next iteration's candidate
            
            front_num += 1
            # Optimization: We can stop if we've assigned ranks to all
            if all(ranks != 0 or n_dom[i] > 0 for i in range(n)) and front_num > 1:
                 # Actually, just break if no more zero-dom individuals
                 if not any(n_dom[i] == 0 and ranks[i] == 0 for i in range(n)):
                     break

        # Assign rank 0 to those not assigned? No, they are dominated.
        # But standard NSGA-II assigns rank to all.
        # The above loop might miss some if logic is slightly off.
        # Let's use a simpler, robust implementation for small N.
        
        # Reset for robust simple implementation
        ranks = np.zeros(n, dtype=int)
        fronts = []
        current_front = set(range(n))
        
        while current_front:
            fronts.append(list(current_front))
            next_front = set()
            for i in current_front:
                for j in range(n):
                    if j in current_front:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        next_front.add(j)
            
            # Remove current front from consideration for next fronts
            # Actually, we need to remove dominated points from the pool of "undominated"
            # Standard algorithm:
            # 1. Find non-dominated set.
            # 2. Remove them.
            # 3. Repeat.
            
            # My current loop logic is flawed for finding subsequent fronts efficiently.
            # Let's stick to the S/n_dom method but ensure it terminates correctly.
            
        # Re-doing Non-dominated sort with a cleaner, standard approach
        ranks = np.zeros(n, dtype=int)
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        front = [set()]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        front_num = 0
        while True:
            current_front = set()
            for i in range(n):
                if n_dom[i] == 0 and ranks[i] == 0:
                    current_front.add(i)
            
            if not current_front:
                break
                
            for i in current_front:
                ranks[i] = front_num
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        # j is ready for next front, but we add it in next iteration
                        pass
            
            front_num += 1
            if not any(n_dom[i] == 0 and ranks[i] == 0 for i in range(n)):
                break

        # Assign remaining unranked (should not happen if logic is correct, but safe fallback)
        for i in range(n):
            if ranks[i] == 0 and n_dom[i] > 0:
                ranks[i] = front_num # They are dominated by previous fronts

        return ranks

    def _crowding_distance(self, F, ranks):
        n = len(F)
        crowd_dist = np.zeros(n)
        n_fronts = max(ranks) + 1
        
        for r in range(n_fronts):
            front_indices = np.where(ranks == r)[0]
            m = len(front_indices)
            if m <= 2:
                crowd_dist[front_indices] = np.inf
                continue
            
            for k in range(self.n_obj):
                # Sort front by objective k
                sorted_indices = front_indices[np.argsort(F[front_indices, k])]
                crowd_dist[sorted_indices[0]] = np.inf
                crowd_dist[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                if f_max == f_min:
                    continue
                
                for i in range(1, m - 1):
                    crowd_dist[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / (f_max - f_min)
        
        return crowd_dist

    def _tournament_select(self, ranks, crowd_dist, i1, i2):
        # i1 wins if it has better rank, or same rank and higher crowding distance
        if ranks[i1] < ranks[i2]:
            return True
        elif ranks[i1] > ranks[i2]:
            return False
        else:
            return crowd_dist[i1] >= crowd_dist[i2]

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = np.copy(p1)
        c2 = np.copy(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                    c2[i] = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta):
        x_mut = np.copy(x)
        for i in range(self.dim):
            if np.random.rand() < 0.9: # Mutation rate
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = (2 * xy + (1.0 - 2 * xy) * u) ** (1.0 / (eta + 1.0))
                    deltaq = (val - 1.0) * (hi[i] - lo[i])
                else:
                    xy = 1.0 - delta2
                    val = (2 * xy + (1.0 - 2 * xy) * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    deltaq = (val - 1.0) * (hi[i] - lo[i])
                
                x_mut[i] += deltaq
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
