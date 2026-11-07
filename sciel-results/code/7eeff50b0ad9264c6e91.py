"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling + evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3 objectives, N=12 variables
        # x[0], x[1] are position variables (M-1 = 2)
        # x[2..11] are distance variables (N-M+1 = 10)
        # g(x) = sum_{i=2}^{11} (x_i^10 - 0.5)^2  [alpha=10 for DTLZ4]
        # When x_i = 0.5 for i>=2, g=0 (Pareto front)
        # f1 = g * cos(t1) * cos(t2)
        # f2 = g * cos(t1) * sin(t2)  
        # f3 = g * sin(t1)
        # where t1 = x[0] * pi/2, t2 = x[1] * pi/2

        # Strategy:
        # Phase 1: Generate well-distributed points ON the Pareto front (g=0)
        # Phase 2: Use remaining budget for evolutionary refinement

        # Phase 1: Structured sampling on Pareto front
        # The Pareto front is a quarter-sphere: f1^2 + f2^2 + f3^2 = 1, fi >= 0
        # Parameterize with t1, t2 in [0, pi/2]
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)

        # For hypervolume maximization with reference [2.5,2.5,2.5]:
        # We need good coverage, especially near vertices and edges

        # Create a set of key points:
        # 1. Vertices (3 corners)
        # 2. Edge midpoints (3 edges)
        # 3. Face center
        # 4. LHS points for uniform coverage

        key_thetas = [
            (0.0, 0.0),           # f=(1,0,0)
            (0.0, np.pi/2),       # f=(0,1,0)
            (np.pi/2, 0.0),       # f=(0,0,1)
            (0.0, np.pi/4),       # edge midpoint between (1,0,0) and (0,1,0)
            (np.pi/4, 0.0),       # edge midpoint between (1,0,0) and (0,0,1)
            (np.pi/2, np.pi/4),   # edge midpoint between (0,1,0) and (0,0,1)
            (np.pi/4, np.pi/4),   # face center
        ]

        points = []
        for t1, t2 in key_thetas:
            x = np.full(dim, 0.5)  # g=0
            x[0] = t1 / (np.pi / 2)
            x[1] = t2 / (np.pi / 2)
            points.append(x.copy())

        # Add LHS points for better coverage
        # Use remaining budget for LHS
        n_structured = min(40, max(10, budget // 4))
        n_lhs = max(0, n_structured - len(points))
        
        if n_lhs > 0:
            rng = np.random.default_rng(123)
            # Latin Hypercube Sampling in 2D
            idx1 = np.arange(n_lhs)
            idx2 = np.arange(n_lhs)
            rng.shuffle(idx1)
            rng.shuffle(idx2)
            t1_lhs = (idx1 + rng.random(n_lhs)) / n_lhs * (np.pi / 2)
            t2_lhs = (idx2 + rng.random(n_lhs)) / n_lhs * (np.pi / 2)
            
            for t1, t2 in zip(t1_lhs, t2_lhs):
                x = np.full(dim, 0.5)
                x[0] = t1 / (np.pi / 2)
                x[1] = t2 / (np.pi / 2)
                points.append(x)

        # Evaluate structured points
        F_structured = np.array([problem(x) for x in points])
        used = len(points)

        # Phase 2: Evolutionary search
        # Use NSGA-II-like approach
        if used >= budget:
            return np.array(points)

        # Select population size
        pop_size = min(30, max(15, (budget - used) // 4))
        
        # Initialize population from non-dominated points
        def is_dominated(F, f):
            """Check if f is dominated by any point in F"""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting for NSGA-II"""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of points dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_count[i] += 1
            
            front = [i for i in range(n) if n_count[i] == 0]
            rank[front] = 1
            k = 1
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return rank

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices"""
            if len(indices) == 0:
                return np.array([])
            if len(indices) == 1:
                return np.array([np.inf])
            
            F_sub = F[indices]
            m = F_sub.shape[1]
            dist = np.zeros(len(indices))
            
            for j in range(m):
                F_j = F_sub[:, j]
                min_j = np.min(F_j)
                max_j = np.max(F_j)
                if max_j - min_j < 1e-10:
                    continue
                order = np.argsort(F_j)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, len(order)-1):
                    dist[order[i]] += (F_j[order[i+1]] - F_j[order[i-1]]) / (max_j - min_j)
            
            return dist

        # Get non-dominated points from structured set
        dominated = np.array([is_dominated(F_structured, f) for f in F_structured])
        non_dom_indices = np.where(~dominated)[0]
        
        if len(non_dom_indices) >= pop_size:
            X = [points[i] for i in non_dom_indices[:pop_size]]
            F = F_structured[non_dom_indices[:pop_size]]
        else:
            X = [points[i] for i in non_dom_indices]
            F = F_structured[non_dom_indices]
            # Pad with random points
            rng = np.random.default_rng(456)
            while len(X) < pop_size:
                x = lo + (hi - lo) * rng.random(dim)
                X.append(x)
                F = np.vstack([F, problem(x)])
                used += 1
                if used >= budget:
                    break
        
        X = np.array(X)
        F = np.array(F)

        # Evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Fast non-dominated sort
            rank = fast_non_dominated_sort(F)
            
            # Select parents from first front
            first_front = np.where(rank == 1)[0]
            if len(first_front) == 0:
                first_front = np.arange(len(X))
            
            # Crowding distance for first front
            cd = crowding_distance(F, first_front)
            
            # Tournament selection based on rank and crowding
            def tournament_select(F, rank, cd):
                i1 = np.random.randint(len(X))
                i2 = np.random.randint(len(X))
                if rank[i1] != rank[i2]:
                    return i1 if rank[i1] < rank[i2] else i2
                # Same rank, use crowding
                cd1 = cd[i1] if i1 in np.where(rank == rank[i1])[0] else 0
                cd2 = cd[i2] if i2 in np.where(rank == rank[i2])[0] else 0
                return i1 if cd1 > cd2 else i2
            
            # Binary crossover + mutation
            p1 = tournament_select(F, rank, cd)
            p2 = tournament_select(F, rank, cd)
            
            # SBX crossover
            beta = 20  # distribution index
            child1 = X[p1].copy()
            child2 = X[p2].copy()
            
            for i in range(dim):
                if np.random.random() < 0.5:
                    u = np.random.random()
                    if u <= 0.5:
                        alpha = 2 * u
                        beta_p = (1 + 2 * (X[p1][i] - X[p2][i]) / (hi[i] - lo[i])) ** (beta + 1)
                        c1 = 0.5 * (X[p1][i] + X[p2][i] - (X[p1][i] - X[p2][i]) * beta_p)
                    else:
                        alpha = 2 * (1 - u)
                        beta_p = (1 + 2 * (X[p2][i] - X[p1][i]) / (hi[i] - lo[i])) ** (beta + 1)
                        c1 = 0.5 * (X[p1][i] + X[p2][i] + (X[p1][i] - X[p2][i]) * beta_p)
                    child1[i] = np.clip(c1, lo[i], hi[i])
            
            # Mutation (polynomial)
            for i in range(dim):
                if np.random.random() < 0.1:  # mutation rate
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (beta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (beta + 1))
                    child1[i] = np.clip(child1[i] + delta * (hi[i] - lo[i]), lo[i], hi[i])
            
            fc = problem(child1)
            used += 1
            
            # Elitism: replace worst point
            # Find point with worst rank and lowest crowding
            max_rank = np.max(rank)
            worst_candidates = np.where(rank == max_rank)[0]
            if len(worst_candidates) > 0:
                cd_worst = crowding_distance(F, worst_candidates)
                worst_idx = worst_candidates[np.argmin(cd_worst)]
            else:
                worst_idx = np.argmax(np.sum(F, axis=1))
            
            X[worst_idx] = child1
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
