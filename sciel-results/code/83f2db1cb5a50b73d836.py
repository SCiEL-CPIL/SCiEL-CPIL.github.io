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
        n = self.dim
        
        # Population size
        pop_size = 20
        
        # Initialize population using LHS for better spread
        X = np.zeros((pop_size, n))
        for j in range(n):
            perm = np.random.permutation(pop_size)
            X[:, j] = (perm + 0.5) / pop_size
        X = lo + (hi - lo) * X
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        self.eval_count += pop_size
        
        # Main Loop
        while self.eval_count < self.budget:
            # Generate offspring
            offspring_X = np.zeros((pop_size, n))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Selection: Binary Tournament on Rank and Crowding
                # We need to calculate ranks and crowding for current X, F
                # To save time, we can approximate or just do it every generation.
                # With N=20, sorting is cheap.
                
                # 1. Non-dominated sorting on X, F
                ranks, fronts = self._non_dominated_sort(F)
                
                # 2. Crowding distance
                cd = self._crowding_distance(F, fronts)
                
                # 3. Select parents
                p1 = self._tournament(X, ranks, cd, 2)
                p2 = self._tournament(X, ranks, cd, 2)
                
                # Crossover: SBX
                child1, child2 = self._sbx(p1, p2, lo, hi)
                
                # Mutation
                child1 = self._polynomial_mutation(child1, lo, hi)
                child2 = self._polynomial_mutation(child2, lo, hi)
                
                # Evaluate child1
                f1 = problem(child1)
                self.eval_count += 1
                offspring_X[i] = child1
                offspring_F[i] = f1
                
                if self.eval_count >= self.budget:
                    break
                    
                # Evaluate child2
                f2 = problem(child2)
                self.eval_count += 1
                if i + 1 < pop_size:
                    offspring_X[i+1] = child2
                    offspring_F[i+1] = f2
                    i += 1 # Skip next index as we filled 2 children
            
            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Non-dominated sorting on combined set
            ranks_all, fronts_all = self._non_dominated_sort(F_all)
            
            # Select next generation
            next_X = np.zeros((pop_size, n))
            next_F = np.zeros((pop_size, self.n_obj))
            
            # Fill by fronts
            current_index = 0
            for front in fronts_all:
                if current_index + len(front) <= pop_size:
                    # Take all
                    next_X[current_index:current_index+len(front)] = X_all[front]
                    next_F[current_index:current_index+len(front)] = F_all[front]
                    current_index += len(front)
                else:
                    # Need to select from this front based on crowding
                    # Calculate crowding distance for this front in the combined set?
                    # Simpler: just take the first ones or sort by crowding.
                    # Let's compute crowding for the combined set to be safe, or just pick arbitrarily if budget is tight.
                    # Given small size, let's compute CD for the combined set.
                    cd_all = self._crowding_distance(F_all, fronts_all)
                    # Sort indices in this front by CD descending
                    front_indices = sorted(front, key=lambda idx: cd_all[idx], reverse=True)
                    needed = pop_size - current_index
                    selected = front_indices[:needed]
                    next_X[current_index:pop_size] = X_all[selected]
                    next_F[current_index:pop_size] = F_all[selected]
                    current_index = pop_size
                    break
            
            X = next_X
            F = next_F
            
        return X

    def _non_dominated_sort(self, F):
        """
        Returns ranks (array of same length as F) and fronts (list of lists of indices).
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        
        S = [[] for _ in range(n)] # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int) # Number of solutions dominating i
        
        for i in range(n):
            for j in range(i+1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        current_front_idx = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = current_front_idx + 1
                        next_front.append(j)
            current_front = next_front
            current_front_idx += 1
            
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        """
        Calculate crowding distance for each solution.
        """
        n = len(F)
        cd = np.zeros(n)
        
        for front in fronts:
            m = len(front)
            if m <= 2:
                for idx in front:
                    cd[idx] = float('inf')
                continue
                
            for k in range(self.n_obj):
                # Sort front by k-th objective
                sorted_front = sorted(front, key=lambda idx: F[idx][k])
                # Assign inf to extremes
                cd[sorted_front[0]] = float('inf')
                cd[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for i in range(1, m-1):
                    cd[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / range_f
                    
        return cd

    def _tournament(self, X, ranks, cd, k=2):
        """
        Binary tournament selection.
        """
        n = len(X)
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        
        # Compare ranks first
        if ranks[idx1] < ranks[idx2]:
            return X[idx1]
        elif ranks[idx2] < ranks[idx1]:
            return X[idx2]
        else:
            # Compare crowding distance
            if cd[idx1] > cd[idx2]:
                return X[idx1]
            else:
                return X[idx2]

    def _sbx(self, p1, p2, lo, hi, eta=20):
        """
        Simulated Binary Crossover.
        """
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2 * u
                        beta = ((1 + 2 * (x1 - lo[i]) / (hi[i] - lo[i])) ** (eta + 1))
                        factor1 = (0.5 * (1 - beta ** (-1) * (alpha - 1) + 1)) # Simplified SBX logic
                        # Standard SBX formula:
                        # c1 = 0.5 * (1 + beta * (x1 - x2) + x1 + x2) ... wait, let's use standard implementation.
                        
                        # Let's use a simpler, robust SBX implementation
                        beta1 = (2 * u) ** (1.0 / (eta + 1.0))
                        beta2 = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        
                        c1[i] = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
                        c2[i] = 0.5 * ((1 + beta2) * x2 + (1 - beta2) * x1)
                    else:
                        alpha = 2 * (1 - u)
                        beta1 = (0.5 / alpha) ** (1.0 / (eta + 1.0))
                        beta2 = (0.5 / (1 - alpha)) ** (1.0 / (eta + 1.0))
                        
                        c1[i] = 0.5 * ((1 + beta1) * x2 + (1 - beta1) * x1)
                        c2[i] = 0.5 * ((1 + beta2) * x1 + (1 - beta2) * x2)
                        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta=20):
        """
        Polynomial Mutation.
        """
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    val = 2 * u
                    delta = (val) ** (1.0 / (eta + 1.0))
                    mut = (x[i] - lo[i]) / (hi[i] - lo[i])
                    mut += delta * (1 - mut) ** eta
                    mut = np.clip(mut, 0, 1)
                    x[i] = lo[i] + (hi[i] - lo[i]) * mut
                else:
                    val = 2 * (1 - u)
                    delta = (val) ** (1.0 / (eta + 1.0))
                    mut = (x[i] - lo[i]) / (hi[i] - lo[i])
                    mut -= delta * (1 - mut) ** eta
                    mut = np.clip(mut, 0, 1)
                    x[i] = lo[i] + (hi[i] - lo[i]) * mut
        return x
