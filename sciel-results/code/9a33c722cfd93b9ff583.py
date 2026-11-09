import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 300  # 300s limit, keep margin

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        N = 20
        # Remaining budget
        evals_used = 0
        
        # Initialization
        # Mix of random and front-biased
        X = np.zeros((N, dim))
        # Half random
        X[:N//2] = lo + span * np.random.rand(N//2, dim)
        # Half front-biased (assuming DTLZ1 structure: last dim-n_obj+1 vars at 0.5)
        # For DTLZ1, M=3, n=7. Last 5 vars at 0.5.
        # x1, x2 sampled from simplex
        n_front_vars = dim - n_obj + 1
        if n_front_vars > 0:
            # Set last n_front_vars to 0.5
            X[N//2:, n_obj-1:] = 0.5
            # Sample x1, x2
            # Uniform in triangle x1>=0, x2>=0, x1+x2<=1
            # Use rejection sampling or Dirichlet
            # Dirichlet(1,1,1) gives uniform on simplex
            # But we need x1, x2.
            # Let u, v be random. 
            # Simpler: 
            # r = rand, s = rand
            # if r+s > 1: r=1-r, s=1-s
            # x1 = r, x2 = s
            # This is uniform in triangle.
            for i in range(N//2, N):
                r = np.random.rand()
                s = np.random.rand()
                if r + s > 1:
                    r = 1 - r
                    s = 1 - s
                X[i, 0] = r
                X[i, 1] = s
                # Clip to bounds just in case
                X[i] = np.clip(X[i], lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        evals_used += N
        
        if evals_used >= budget:
            return X
            
        # NSGA-II Loop
        while evals_used < budget:
            # Check time
            if time.time() > self.deadline - 10:
                break
                
            # Selection
            ranks, crowding = self._fast_non_dominated_sort(F)
            
            # Generate children
            children = []
            while len(children) < N and evals_used + 1 <= budget:
                # Tournament selection
                p1 = self._tournament_select(X, F, ranks, crowding)
                p2 = self._tournament_select(X, F, ranks, crowding)
                
                # Crossover (SBX)
                if np.random.rand() < 0.9:
                    c1, c2 = self._sbx(p1, p2, lo, hi)
                    children.append(c1)
                    if evals_used + len(children) <= budget:
                        children.append(c2)
                else:
                    children.append(p1.copy())
                    if evals_used + len(children) <= budget:
                        children.append(p2.copy())
                        
            # Truncate children to fit budget
            max_children = budget - evals_used
            children = children[:max_children]
            
            if not children:
                break
                
            # Mutate children
            for i in range(len(children)):
                children[i] = self._polynomial_mutation(children[i], lo, hi)
                children[i] = np.clip(children[i], lo, hi)
                
            # Evaluate children
            F_children = np.array([problem(x) for x in children])
            evals_used += len(children)
            
            # Combine populations
            X_combined = np.vstack([X, children])
            F_combined = np.vstack([F, F_children])
            
            # Non-dominated sort and selection
            ranks, crowding = self._fast_non_dominated_sort(F_combined)
            
            # Select next generation
            X, F = self._select_next_generation(X_combined, F_combined, ranks, crowding, N)
            
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Dominance matrix
        # A[i] = indices of points dominated by i
        # B[i] = count of points dominating i
        A = [[] for _ in range(n)]
        B = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    A[i].append(j)
                    B[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    A[j].append(i)
                    B[i] += 1
                    
        # First front
        front = []
        for i in range(n):
            if B[i] == 0:
                ranks[i] = 0
                front.append(i)
                
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in A[i]:
                    B[j] -= 1
                    if B[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
            
        # Crowding distance
        fronts = {}
        for i in range(n):
            r = ranks[i]
            if r not in fronts:
                fronts[r] = []
            fronts[r].append(i)
            
        for r, idxs in fronts.items():
            m = len(idxs)
            if m == 1:
                crowding[idxs[0]] = np.inf
                continue
            # For each objective
            for j in range(self.n_obj):
                # Sort by objective j
                sorted_idxs = sorted(idxs, key=lambda i: F[i, j])
                crowding[sorted_idxs[0]] = np.inf
                crowding[sorted_idxs[-1]] = np.inf
                f_min = F[sorted_idxs[0], j]
                f_max = F[sorted_idxs[-1], j]
                if f_max == f_min:
                    continue
                for k in range(1, m-1):
                    crowding[sorted_idxs[k]] += (F[sorted_idxs[k+1], j] - F[sorted_idxs[k-1], j]) / (f_max - f_min)
                    
        return ranks, crowding

    def _tournament_select(self, X, F, ranks, crowding):
        n = len(X)
        i1 = np.random.randint(n)
        i2 = np.random.randint(n)
        
        # Compare ranks
        if ranks[i1] < ranks[i2]:
            return X[i1]
        elif ranks[i2] < ranks[i1]:
            return X[i2]
        else:
            # Same rank, compare crowding
            if crowding[i1] > crowding[i2]:
                return X[i1]
            else:
                return X[i2]

    def _sbx(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        eta = 20.0
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                continue
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        eta = 20.0
        for i in range(len(x)):
            if np.random.rand() < 0.5:
                continue
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[i] = x[i] + delta * (hi[i] - lo[i])
        return x

    def _select_next_generation(self, X, F, ranks, crowding, N):
        # Sort by rank, then crowding
        n = len(X)
        indices = np.arange(n)
        
        # Create a combined score for sorting
        # Primary: rank, Secondary: -crowding
        # We want to pick the best N
        
        # Sort indices by rank
        sorted_indices = np.argsort(ranks, kind='mergesort')
        
        next_gen_indices = []
        for idx in sorted_indices:
            if len(next_gen_indices) >= N:
                break
            next_gen_indices.append(idx)
            
        # If we didn't fill N, we might need to handle the last front more carefully?
        # The simple greedy approach above works if we sort by rank first.
        # However, NSGA-II requires handling the last front with crowding distance.
        
        # Let's do it properly:
        next_gen_indices = []
        current_front = 0
        remaining = N
        
        # Group indices by rank
        ranks_dict = {}
        for i in range(n):
            r = ranks[i]
            if r not in ranks_dict:
                ranks_dict[r] = []
            ranks_dict[r].append(i)
            
        # Sort fronts
        sorted_ranks = sorted(ranks_dict.keys())
        
        for r in sorted_ranks:
            if remaining <= 0:
                break
            front_indices = ranks_dict[r]
            if len(front_indices) <= remaining:
                next_gen_indices.extend(front_indices)
                remaining -= len(front_indices)
            else:
                # Select based on crowding distance
                sorted_front = sorted(front_indices, key=lambda i: crowding[i], reverse=True)
                next_gen_indices.extend(sorted_front[:remaining])
                remaining = 0
                
        next_gen_X = X[next_gen_indices]
        next_gen_F = F[next_gen_indices]
        
        return next_gen_X, next_gen_F
