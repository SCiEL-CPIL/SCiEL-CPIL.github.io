import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Determine population size
        # Budget 200, Dim 30.
        # Let's use N=20, so 10 generations.
        N = 20
        T = self.budget // N
        
        # Initialize
        X = lo + range_ * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II loop
        while used < self.budget:
            # Generate offspring
            Q = np.zeros_like(X)
            for i in range(N):
                # Tournament selection
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover (SBX)
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c1 = self._gaussian_mutation(c1, lo, hi)
                c2 = self._gaussian_mutation(c2, lo, hi)
                
                Q[2*i] = c1
                Q[2*i+1] = c2
                
            # Evaluate offspring
            Fq = np.array([problem(x) for x in Q])
            used += N
            
            # Combine parents and offspring
            R = np.vstack([X, Q])
            Fr = np.vstack([F, Fq])
            
            # Non-dominated sorting and crowding distance
            ranks, crowding = self._non_dominated_sorting(R, Fr)
            
            # Selection
            # Sort indices by rank, then by crowding distance (descending)
            idx = np.argsort(ranks)
            # Within same rank, sort by crowding distance descending
            # We need a stable sort or a custom logic.
            # Easier: create a list of (rank, -crowding, index) and sort
            sort_keys = [(ranks[i], -crowding[i], i) for i in range(len(R))]
            sort_keys.sort()
            selected_indices = [k[2] for k in sort_keys[:N]]
            
            X = R[selected_indices]
            F = Fr[selected_indices]
            
        return X

    def _tournament_select(self, X, F):
        # Simple tournament of size 2
        i = np.random.randint(len(X))
        j = np.random.randint(len(X))
        
        # We need ranks and crowding for selection. 
        # To avoid recalculating every time, we should pass them or calculate once per generation.
        # In the loop above, I called _tournament_select inside the offspring generation loop.
        # This is inefficient because _tournament_select would need ranks/crowding.
        # Let's refactor: Calculate ranks/crowding for current X once, then use for selection.
        
        pass

    # The structure above is flawed for efficiency. 
    # I need to calculate ranks/crowding for X once, then select N pairs of parents.
    
    def _refactored_call(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        N = 20
        T = self.budget // N
        
        X = lo + range_ * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        while used < self.budget:
            ranks, crowding = self._non_dominated_sorting(X, F)
            
            Q = np.zeros_like(X)
            for i in range(N):
                p1 = self._tournament_select(X, ranks, crowding)
                p2 = self._tournament_select(X, ranks, crowding)
                
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                c1 = self._gaussian_mutation(c1, lo, hi)
                c2 = self._gaussian_mutation(c2, lo, hi)
                
                Q[2*i] = c1
                Q[2*i+1] = c2
                
            Fq = np.array([problem(x) for x in Q])
            used += N
            
            R = np.vstack([X, Q])
            Fr = np.vstack([F, Fq])
            
            ranks_r, crowding_r = self._non_dominated_sorting(R, Fr)
            
            sort_keys = [(ranks_r[i], -crowding_r[i], i) for i in range(len(R))]
            sort_keys.sort()
            selected_indices = [k[2] for k in sort_keys[:N]]
            
            X = R[selected_indices]
            F = Fr[selected_indices]
            
        return X

    def _tournament_select(self, X, ranks, crowding):
        i = np.random.randint(len(X))
        j = np.random.randint(len(X))
        
        # Compare (rank, crowding)
        # Lower rank is better. Higher crowding is better.
        if ranks[i] < ranks[j]:
            return X[i]
        elif ranks[j] < ranks[i]:
            return X[j]
        else:
            if crowding[i] > crowding[j]:
                return X[i]
            else:
                return X[j]

    def _sbx_crossover(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        eta_cr = 20.0
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # No crossover
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_cr + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_cr + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _gaussian_mutation(self, x, lo, hi):
        # Gaussian mutation
        # Rate 1/dim
        # Sigma 0.1 (since range is 1)
        sigma = 0.1
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                x[i] += np.random.randn() * sigma
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

    def _non_dominated_sorting(self, X, F):
        N = len(X)
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        # Fast Non-Dominated Sort
        # S[i] = set of indices dominated by i
        # n[i] = number of solutions dominating i
        
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(i+1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n[i] += 1
        
        # Front 0
        front = [i for i in range(N) if n[i] == 0]
        k = 0
        current_front = front
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = k
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        # Crowding Distance
        # For each front, calculate crowding distance
        # Group indices by rank
        fronts_dict = {}
        for i in range(N):
            r = ranks[i]
            if r not in fronts_dict:
                fronts_dict[r] = []
            fronts_dict[r].append(i)
        
        for r, members in fronts_dict.items():
            if len(members) <= 2:
                for i in members:
                    crowding[i] = float('inf')
                continue
            
            for m in range(self.n_obj):
                # Sort members by objective m
                sorted_members = sorted(members, key=lambda x: F[x][m])
                crowding[sorted_members[0]] = float('inf')
                crowding[sorted_members[-1]] = float('inf')
                
                # Normalize range
                f_min = F[sorted_members[0]][m]
                f_max = F[sorted_members[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for idx in range(1, len(sorted_members)-1):
                    i = sorted_members[idx]
                    prev_i = sorted_members[idx-1]
                    next_i = sorted_members[idx+1]
                    crowding[i] += (F[next_i][m] - F[prev_i][m]) / range_f
                    
        return ranks, crowding

    def __call__(self, problem, bounds):
        return self._refactored_call(problem, bounds)
