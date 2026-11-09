import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        # Initialize population
        X = np.random.uniform(lo, hi, (pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            front = [0]
            current_front = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # Check if q dominates p
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front.append(len(next_front))
                current_front = next_front
            return rank, front
        
        def crowding_distance(F, rank, front_idx):
            # F: array of objective values
            # rank: array of ranks for all points
            # front_idx: which front to calculate distance for
            n = len(F)
            dist = np.zeros(n)
            front_points = [i for i in range(n) if rank[i] == front_idx]
            m = len(front_points)
            if m == 0: return dist
            if m == 1:
                dist[front_points[0]] = float('inf')
                return dist
            
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_indices = sorted(front_points, key=lambda x: F[x][k])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                range_val = F[sorted_indices[-1]][k] - F[sorted_indices[0]][k]
                if range_val == 0: continue
                for i in range(1, m-1):
                    dist[sorted_indices[i]] += (F[sorted_indices[i+1]][k] - F[sorted_indices[i-1]][k]) / range_val
            return dist

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            # Simple: create same number of offspring as population
            for _ in range(pop_size):
                if np.random.rand() < 0.9: # Crossover
                    p1_idx = np.random.randint(pop_size)
                    p2_idx = np.random.randint(pop_size)
                    while p2_idx == p1_idx:
                        p2_idx = np.random.randint(pop_size)
                    x1 = X[p1_idx]
                    x2 = X[p2_idx]
                    # SBX Crossover
                    u = np.random.rand(self.dim)
                    beta = np.where(u <= 0.5, (2*u)**(1/21), (2*(1-u))**(1/21))
                    child1 = 0.5 * ((1+beta)*x1 + (1-beta)*x2)
                    child2 = 0.5 * ((1-beta)*x1 + (1+beta)*x2)
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    p1_idx = np.random.randint(pop_size)
                    offspring_X.append(X[p1_idx].copy())
            
            # We might generate too many, so we trim to pop_size
            if len(offspring_X) > pop_size:
                offspring_X = offspring_X[:pop_size]
            
            # Mutation
            for i in range(len(offspring_X)):
                if np.random.rand() < 0.9:
                    x = offspring_X[i]
                    # Polynomial Mutation approximation or Gaussian
                    # Gaussian is simpler and effective
                    sigma = 0.2 * (hi - lo)
                    x += np.random.normal(0, sigma, self.dim)
                    x = np.clip(x, lo, hi)
                    offspring_X[i] = x
            
            # Evaluate Offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_F)
            
            # Combine
            X_total = np.vstack([X, offspring_X])
            F_total = np.vstack([F, offspring_F])
            
            # Selection
            rank, front_sizes = non_dominated_sort(F_total)
            
            new_X = []
            new_F = []
            for i in range(len(front_sizes)):
                front_size = front_sizes[i]
                if len(new_X) + front_size <= pop_size:
                    # Take all
                    for j in range(len(F_total)):
                        if rank[j] == i:
                            new_X.append(X_total[j])
                            new_F.append(F_total[j])
                else:
                    # Need to select some based on crowding
                    front_points = [j for j in range(len(F_total)) if rank[j] == i]
                    # Calculate crowding distance for this front
                    # We need a helper to get dist for a specific subset
                    # Re-using crowding_distance logic but on subset indices
                    dist = np.zeros(len(F_total))
                    m = len(front_points)
                    if m > 1:
                        for k in range(self.n_obj):
                            sorted_indices = sorted(front_points, key=lambda x: F_total[x][k])
                            dist[sorted_indices[0]] = float('inf')
                            dist[sorted_indices[-1]] = float('inf')
                            range_val = F_total[sorted_indices[-1]][k] - F_total[sorted_indices[0]][k]
                            if range_val != 0:
                                for idx in range(1, m-1):
                                    dist[sorted_indices[idx]] += (F_total[sorted_indices[idx+1]][k] - F_total[sorted_indices[idx-1]][k]) / range_val
                    # Sort by dist descending
                    selected = sorted(front_points, key=lambda x: dist[x], reverse=True)
                    need = pop_size - len(new_X)
                    for j in selected[:need]:
                        new_X.append(X_total[j])
                        new_F.append(F_total[j])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
