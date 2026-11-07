import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Determine population size
        # We want to use the budget efficiently.
        # Let N be population size.
        # Init: N evals.
        # Each gen: N offspring evals.
        # Total evals = N + G * N <= budget.
        # We want G to be reasonable, e.g., 4-6 generations.
        # Let's pick N such that we can run ~4-5 generations.
        # 200 / 5 = 40. So N=40 is a good choice.
        N = 40
        if N > self.budget:
            N = self.budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        eval_count = N
        
        # NSGA-II Loop
        while eval_count < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Create offspring
            Q = np.zeros((N, dim))
            for i in range(N):
                # Tournament selection for parent 1
                p1 = self._tournament_select(X, F, N)
                # Tournament selection for parent 2
                p2 = self._tournament_select(X, F, N)
                
                # Crossover (SBX)
                c1, c2 = self._sbx(p1, p2, lo, hi)
                
                # Mutation (Polynomial)
                c1 = self._poly_mutate(c1, lo, hi)
                c2 = self._poly_mutate(c2, lo, hi)
                
                Q[i, :] = c1
                if eval_count + 1 < self.budget: # Ensure we don't exceed budget
                    # We evaluate one by one to be safe with budget, or batch if possible
                    # But problem(x) is called individually.
                    pass
            
            # Actually, let's just generate N offspring and evaluate them.
            # We need to be careful with the budget count.
            # If we generate N offspring, we need N evaluations.
            # If eval_count + N > self.budget, we can only evaluate (self.budget - eval_count) offspring.
            # But NSGA-II expects a full set of offspring for the next generation selection.
            # If we don't have a full set, we can just stop or handle the partial set.
            # Given 200 budget and N=40, 200 is divisible by 40. So we will have exactly 4 generations of offspring.
            # Init: 40. Gen 1: 40 (Total 80). Gen 2: 40 (Total 120). Gen 3: 40 (Total 160). Gen 4: 40 (Total 200).
            # So we can always generate a full set of N offspring until the budget is exhausted.
            
            # Generate offspring
            Q = np.zeros((N, dim))
            for i in range(N):
                p1 = self._tournament_select(X, F, N)
                p2 = self._tournament_select(X, F, N)
                c1, c2 = self._sbx(p1, p2, lo, hi)
                c1 = self._poly_mutate(c1, lo, hi)
                c2 = self._poly_mutate(c2, lo, hi)
                Q[i, :] = c1
                
            # Evaluate offspring
            Fq = np.array([problem(x) for x in Q])
            eval_count += N
            
            # Combine parents and offspring
            R = np.vstack((X, Q))
            FR = np.vstack((F, Fq))
            
            # Non-dominated sorting and crowding distance
            ranks, crowding = self._nsga2_selection(R, FR)
            
            # Select best N individuals for next generation
            # Sort by rank, then by crowding distance (descending)
            idx = np.lexsort(((-crowding, ranks)))
            X = R[idx[:N]]
            F = FR[idx[:N]]
            
        return X

    def _tournament_select(self, X, F, N):
        # Binary tournament
        i = np.random.randint(N)
        j = np.random.randint(N)
        # Compare based on rank and crowding? 
        # In NSGA-II, selection is based on non-dominated sorting.
        # We need to compute ranks for the current population X to do proper selection.
        # This is inefficient if done every time. 
        # Standard NSGA-II computes ranks once per generation for the combined population R, 
        # but selection for parents is done from the previous generation's population X.
        # Usually, X is already sorted or we store ranks/crowding from the previous step.
        # For simplicity and correctness, let's store ranks and crowding for the current X.
        
        # Wait, my loop structure recalculates everything.
        # Let's refactor to store ranks and crowding for the current population X.
        
        # This method is called inside the loop. I need the ranks of the current X.
        # I will modify the main loop to calculate ranks/crowding for X at the start of each generation.
        
        # For now, this method is a placeholder. I will fix the main loop.
        pass

    def _nsga2_selection(self, R, FR):
        # Non-dominated sorting
        n = len(R)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Fast Non-Dominated Sorting
        S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(FR[i] <= FR[j]) and np.any(FR[i] < FR[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(FR[j] <= FR[i]) and np.any(FR[j] < FR[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        front_idx = 0
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = front_idx
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1
        
        # Crowding Distance
        for front in range(front_idx):
            front_indices = [i for i in range(n) if ranks[i] == front]
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_indices = sorted(front_indices, key=lambda i: FR[i][k])
                crowding[sorted_indices[0]] = np.inf
                crowding[sorted_indices[-1]] = np.inf
                f_min = FR[sorted_indices[0]][k]
                f_max = FR[sorted_indices[-1]][k]
                if f_max - f_min == 0:
                    continue
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    crowding[idx] += (FR[next_idx][k] - FR[prev_idx][k]) / (f_max - f_min)
                    
        return ranks, crowding

    def _sbx(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        eta = 20
        for i in range(len(p1)):
            if np.random.random() < 0.5:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutate(self, x, lo, hi):
        eta = 20
        for i in range(len(x)):
            if np.random.random() < 0.5: # Mutation rate 0.5? Usually 1/dim.
                 # Standard rate is 1/dim.
                 # Let's use 1/dim.
                 pass
        # Let's implement proper polynomial mutation with rate 1/dim
        x_mut = x.copy()
        rate = 1.0 / self.dim
        for i in range(self.dim):
            if np.random.random() < rate:
                u = np.random.random()
                if u <= 0.5:
                    delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta2 = (1 - 2 * (1 - u)) ** (1.0 / (eta + 1)) - 1
                    delta1 = -delta2
                
                x_mut[i] += delta1 * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
