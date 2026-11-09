import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 10
        # Initialize population using LHS for good spread
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[:, j] = (perm + 0.5) / pop_size
        X = lo + (hi - lo) * X
        
        F = np.array([problem(x) for x in X])
        self.eval_count += pop_size
        
        # Main loop
        while self.eval_count < self.budget:
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Pick 2 parents
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Blend Crossover (BLX-0.5 equivalent)
                x_off = p1 + np.random.rand(self.dim) * (p2 - p1)
                
                # Mutation
                x_off += np.random.randn(self.dim) * 0.1 * (hi - lo)
                
                offspring_X[i] = np.clip(x_off, lo, hi)
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X])
            self.eval_count += pop_size
            
            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])
            
            # Selection: Non-dominated sorting + Crowding distance
            # Simple approximation: Keep the best 'pop_size' based on dominance and diversity
            # For simplicity and speed, we can use a greedy selection if full NSGA-II is too long.
            # But let's try to implement a fast non-dominated sort.
            
            ranks = self._non_dominated_sort(F_all)
            
            # Sort by rank, then by crowding distance (descending)
            # We need crowding distance.
            # To keep code short, we might use a simpler diversity metric or just random selection from top ranks.
            # Actually, let's implement a simplified selection:
            # 1. Find non-dominated set (Rank 0).
            # 2. If len(Rank 0) >= pop_size, select pop_size from Rank 0 using crowding distance.
            # 3. Else, take all Rank 0, then Rank 1, etc.
            
            selected_indices = []
            current_rank = 0
            while len(selected_indices) < pop_size:
                # Get indices with current rank
                rank_indices = np.where(ranks == current_rank)[0]
                
                if len(rank_indices) + len(selected_indices) <= pop_size:
                    selected_indices.extend(rank_indices)
                else:
                    # Need to select from this rank using crowding distance
                    F_rank = F_all[rank_indices]
                    # Calculate crowding distance for this subset
                    cd = self._crowding_distance(F_rank)
                    # Sort by cd descending
                    order = np.argsort(-cd)
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(rank_indices[order[:needed]])
                    break
                
                current_rank += 1
                if current_rank > len(ranks):
                    break
            
            X = X_all[selected_indices]
            F = F_all[selected_indices]
        
        return np.array([x for x in X]) # Return value ignored, but must return something

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [set() for _ in range(n)]
        n_dom = [0] * n
        ranks = [0] * n
        current_front = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(F[i], F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
            
            if n_dom[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        current_rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            current_front = next_front
            current_rank += 1
            
        return ranks

    def _dominates(self, f1, f2):
        # f1 dominates f2 if f1 <= f2 in all obj and < in at least one
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(self.n_obj):
            # Sort by objective m
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F[order[-1], m] - F[order[0], m]
            if f_range == 0:
                continue
            for i in range(1, n-1):
                cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / f_range
        return cd
