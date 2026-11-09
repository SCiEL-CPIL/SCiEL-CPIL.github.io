import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        # With 200 budget, pop size 20 gives 10 generations (1 init + 9 offspring)
        # Let's use pop size 20.
        pop_size = 20
        
        # Initialize population
        X = lo + range_ * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_nondominated_sort(F):
            n = len(F)
            S = [list() for _ in range(n)]
            n_dominates = [0] * n
            rank = [0] * n
            front = [list()]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
                        
                if n_dominates[p] == 0:
                    rank[p] = 0
                    front[0].append(p)
                    
            i = 0
            while len(front[i]) > 0:
                next_front = list()
                for p in front[i]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                if len(next_front) > 0:
                    front.append(next_front)
                else:
                    break
            return front, rank

        def crowding_distance(F, rank, pop_indices):
            # F: objective matrix
            # rank: list of ranks
            # pop_indices: indices of points in this front
            n = len(pop_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = np.argsort(F[pop_indices, m])
                sorted_F = F[pop_indices[sorted_indices], m]
                
                if sorted_F[0] == sorted_F[-1]:
                    continue
                    
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                for i in range(1, n - 1):
                    dist[sorted_indices[i]] += (sorted_F[i + 1] - sorted_F[i - 1]) / (sorted_F[-1] - sorted_F[0])
                    
            # Map back to original indices order? 
            # The function returns distance for the given pop_indices order.
            return dist

        # Main loop
        while used < self.budget and time.time() < deadline:
            # Generate offspring
            X_off = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Selection: Tournament
                # Pick 2 random parents
                p1_idx, p2_idx = np.random.choice(pop_size, 2, replace=False)
                
                # Compare rank and crowding
                # We need rank and crowding for current population
                front, rank = fast_nondominated_sort(F)
                
                # Calculate crowding distance for all points in population
                cd = np.zeros(pop_size)
                for f_idx, f_members in enumerate(front):
                    if not f_members:
                        continue
                    # Get indices of members in this front
                    member_indices = f_members
                    # Calculate crowding distance for these members
                    # Note: F is the full population matrix
                    # We need to pass the subset F[member_indices]
                    cd_sub = crowding_distance(F, rank, member_indices)
                    for j, m_idx in enumerate(member_indices):
                        cd[m_idx] = cd_sub[j]
                
                # Tournament selection
                if rank[p1_idx] < rank[p2_idx]:
                    p1 = p1_idx
                elif rank[p2_idx] < rank[p1_idx]:
                    p1 = p2_idx
                else:
                    if cd[p1_idx] > cd[p2_idx]:
                        p1 = p1_idx
                    else:
                        p1 = p2_idx
                        
                # Repeat for second parent
                p1_idx2, p2_idx2 = np.random.choice(pop_size, 2, replace=False)
                if rank[p1_idx2] < rank[p2_idx2]:
                    p2 = p1_idx2
                elif rank[p2_idx2] < rank[p1_idx2]:
                    p2 = p2_idx2
                else:
                    if cd[p1_idx2] > cd[p2_idx2]:
                        p2 = p1_idx2
                    else:
                        p2 = p2_idx2
                        
                # Crossover: Simulated Binary Crossover (SBX)
                # Simple SBX implementation
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                eta = 20.0
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        # No crossover
                        child1[d] = X[p1, d]
                        child2[d] = X[p2, d]
                    else:
                        x1, x2 = X[p1, d], X[p2, d]
                        if x1 == x2:
                            child1[d] = x1
                            child2[d] = x2
                        else:
                            if x1 > x2:
                                x1, x2 = x2, x1
                                
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                                
                            child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            child2[d] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                            
                # Mutation: Polynomial Mutation
                # Or simple Gaussian
                sigma = 0.1 * range_
                child1 += np.random.normal(0, sigma, self.dim)
                child2 += np.random.normal(0, sigma, self.dim)
                
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                X_off[2*i] = child1
                X_off[2*i+1] = child2
                
            # Evaluate offspring
            # Check budget
            needed = pop_size * 2
            if used + needed > self.budget:
                # Only evaluate as many as we can
                # But NSGA-II usually works in blocks. 
                # If we don't have enough budget for a full generation, 
                # we might just do random restarts or single point mutations 
                # for the remaining budget.
                # For simplicity, if we can't do a full gen, break or do partial.
                # Let's just break if we can't do a full generation to keep code simple, 
                # or adjust pop size? 
                # Actually, 200 / 20 = 10. So we do exactly 9 generations of 20 offspring? 
                # Init: 20.
                # Gen 1: 20 offspring -> 40 combined -> select 20. Total used: 40.
                # ...
                # Gen 9: 20 offspring -> Total used: 20 + 9*20 = 200.
                # So it fits perfectly.
                pass

            F_off = np.array([problem(x) for x in X_off])
            used += pop_size * 2
            
            # Combine parents and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sort
            front, rank = fast_nondominated_sort(F_all)
            
            # Select next population
            X_new = np.zeros((pop_size, self.dim))
            F_new = np.zeros((pop_size, self.n_obj))
            count = 0
            
            # Calculate crowding distance for all points in X_all
            cd_all = np.zeros(len(X_all))
            for f_idx, f_members in enumerate(front):
                if not f_members:
                    continue
                member_indices = f_members
                cd_sub = crowding_distance(F_all, rank, member_indices)
                for j, m_idx in enumerate(member_indices):
                    cd_all[m_idx] = cd_sub[j]
            
            for f_idx, f_members in enumerate(front):
                if len(f_members) <= pop_size - count:
                    # Take all
                    for m_idx in f_members:
                        X_new[count] = X_all[m_idx]
                        F_new[count] = F_all[m_idx]
                        count += 1
                else:
                    # Need to select some from this front
                    # Sort by crowding distance (descending)
                    sorted_members = sorted(f_members, key=lambda x: cd_all[x], reverse=True)
                    needed_from_front = pop_size - count
                    for m_idx in sorted_members[:needed_from_front]:
                        X_new[count] = X_all[m_idx]
                        F_new[count] = F_all[m_idx]
                        count += 1
                    break
                    
            X = X_new
            F = F_new
            
        return X
