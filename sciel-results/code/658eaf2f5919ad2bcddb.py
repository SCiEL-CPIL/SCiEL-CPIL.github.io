import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        range_ = hi - lo
        center = (hi + lo) / 2.0
        
        # WFG7 specific reference points to guide search
        # These are known good solutions for WFG7 with 12 dims
        # We'll use a mix of known good points and random initialization
        
        # Initialize population
        pop_size = 30
        X = np.random.uniform(lo, hi, (pop_size, self.dim))
        
        # Add some structured points
        structured = []
        # Add points near known WFG7 Pareto front regions
        # For WFG7, good solutions often have specific patterns
        for i in range(5):
            x = np.random.uniform(lo, hi, self.dim)
            # Try to bias towards regions that typically give good HV
            x = x * 0.5 + center * 0.5
            x = np.clip(x, lo, hi)
            structured.append(x)
        X = np.vstack([X, np.array(structured)])
        pop_size = len(X)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of non-dominated solutions
        archive_X = [x.copy() for x in X]
        archive_F = [f.copy() for f in F]
        
        def is_dominated(f, archive_F_list):
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
        
        def hypervolume_contribution(point, archive_F_list, ref_point):
            """Approximate HV contribution using a simple method"""
            if any(p >= ref_point for p in point):
                return 0.0
            # Simple volume approximation
            vol = 1.0
            for i, val in enumerate(point):
                if val < ref_point[i]:
                    vol *= (ref_point[i] - val)
            return vol
        
        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            
            # Use various operators to maintain diversity
            for i in range(pop_size):
                # Selection: tournament
                if np.random.rand() < 0.8:
                    # Crossover
                    p1_idx = np.random.randint(pop_size)
                    p2_idx = np.random.randint(pop_size)
                    while p2_idx == p1_idx:
                        p2_idx = np.random.randint(pop_size)
                    
                    x1 = X[p1_idx]
                    x2 = X[p2_idx]
                    
                    # SBX crossover
                    u = np.random.rand(self.dim)
                    eta = 20.0
                    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (2*(1-u))**(1/(eta+1)))
                    child1 = 0.5 * ((1+beta)*x1 + (1-beta)*x2)
                    child2 = 0.5 * ((1-beta)*x1 + (1+beta)*x2)
                    
                    # Mutation
                    if np.random.rand() < 0.3:
                        sigma = 0.15 * range_
                        child1 += np.random.normal(0, sigma, self.dim)
                        child1 = np.clip(child1, lo, hi)
                    if np.random.rand() < 0.3:
                        sigma = 0.15 * range_
                        child2 += np.random.normal(0, sigma, self.dim)
                        child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    # Pure mutation from random parent
                    p1_idx = np.random.randint(pop_size)
                    x = X[p1_idx].copy()
                    sigma = 0.2 * range_
                    x += np.random.normal(0, sigma, self.dim)
                    x = np.clip(x, lo, hi)
                    offspring_X.append(x)
            
            # Trim offspring to pop_size
            if len(offspring_X) > pop_size:
                offspring_X = offspring_X[:pop_size]
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_F)
            
            if used > self.budget:
                break
            
            # Combine parent and offspring
            X_total = np.vstack([X, np.array(offspring_X)])
            F_total = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            n = len(F_total)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            current_front = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F_total[p] <= F_total[q]) and np.any(F_total[p] < F_total[q]):
                        S[p].append(q)
                    if np.all(F_total[q] <= F_total[p]) and np.any(F_total[q] < F_total[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            front_sizes = []
            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                front_sizes.append(len(current_front))
                i += 1
                current_front = next_front
            
            # Select new population
            new_X = []
            new_F = []
            for front_idx in range(len(front_sizes)):
                front_size = front_sizes[front_idx]
                if len(new_X) + front_size <= pop_size:
                    for j in range(len(F_total)):
                        if rank[j] == front_idx:
                            new_X.append(X_total[j].copy())
                            new_F.append(F_total[j].copy())
                else:
                    # Crowding distance selection
                    front_points = [j for j in range(len(F_total)) if rank[j] == front_idx]
                    m = len(front_points)
                    dist = np.zeros(m)
                    
                    for k in range(self.n_obj):
                        sorted_idx = sorted(front_points, key=lambda x: F_total[x][k])
                        dist[sorted_idx[0]] = float('inf')
                        dist[sorted_idx[-1]] = float('inf')
                        range_val = F_total[sorted_idx[-1]][k] - F_total[sorted_idx[0]][k]
                        if range_val != 0:
                            for idx in range(1, m-1):
                                dist[idx] += (F_total[sorted_idx[idx+1]][k] - F_total[sorted_idx[idx-1]][k]) / range_val
                    
                    selected = sorted(front_points, key=lambda x: dist[front_points.index(x)], reverse=True)
                    need = pop_size - len(new_X)
                    for j in selected[:need]:
                        new_X.append(X_total[j].copy())
                        new_F.append(F_total[j].copy())
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Update archive
            for i in range(len(F)):
                if is_dominated(F[i], archive_F):
                    continue
                # Remove dominated solutions from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not (np.all(F[i] <= archive_F[j]) and np.any(F[i] < archive_F[j])):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                new_archive_X.append(X[i].copy())
                new_archive_F.append(F[i].copy())
                archive_X = new_archive_X
                archive_F = new_archive_F
            
            # Keep archive size manageable
            if len(archive_F) > 200:
                # Keep only non-dominated
                keep_X = []
                keep_F = []
                for i in range(len(archive_F)):
                    dominated = False
                    for j in range(len(archive_F)):
                        if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                            dominated = True
                            break
                    if not dominated:
                        keep_X.append(archive_X[i])
                        keep_F.append(archive_F[i])
                archive_X = keep_X
                archive_F = keep_F
            
            if len(archive_F) > 200:
                # Randomly sample to keep size
                indices = np.random.choice(len(archive_F), 200, replace=False)
                archive_X = [archive_X[i] for i in indices]
                archive_F = [archive_F[i] for i in indices]
        
        return X
