"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Fixed seed for reproducibility as per task description
        rng = np.random.default_rng(0)
        
        # Population size: 50 is a good balance for budget 200 (4 generations)
        pop_size = 50
        if budget < 100:
            pop_size = max(10, budget // 2)
            
        # Initialize population
        X = lo + rng.random((pop_size, dim)) * range_
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # NSGA-II Parameters
        cx_prob = 0.9
        mut_prob = 0.1
        eta_c = 20.0
        eta_m = 20.0
        
        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sort.
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = np.zeros(n, dtype=int) # n_d[i] = number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    # j dominates i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_d[i] == 0]
            for i in current_front:
                rank[i] = 0
            fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for solutions in a front.
            Returns array of distances.
            """
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            m = len(front_indices)
            n_obj = F.shape[1]
            dist = np.zeros(m)
            
            for k in range(n_obj):
                f_vals = F[front_indices, k]
                sorted_idx = np.argsort(f_vals)
                sorted_f = f_vals[sorted_idx]
                
                if sorted_f[-1] - sorted_f[0] == 0:
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    continue
                    
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (sorted_f[i+1] - sorted_f[i-1]) / (sorted_f[-1] - sorted_f[0])
                    
            return dist

        def sbx_crossover(x1, x2, rng):
            """Simulated Binary Crossover"""
            if rng.random() > cx_prob:
                return x1, x2
            
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(dim):
                if rng.random() > 0.5:
                    continue
                
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                    
                beta = 1.0
                if x1[i] <= x2[i]:
                    if x1[i] == 0:
                        beta = 1.0
                    else:
                        beta = (x1[i] / x2[i]) ** (1.0 / (eta_c + 1.0))
                else:
                    if x2[i] == 0:
                        beta = 1.0
                    else:
                        beta = (x2[i] / x1[i]) ** (1.0 / (eta_c + 1.0))
                
                # Calculate beta_q1
                if rng.random() * 2 <= 1.0 / (beta + 1.0):
                    beta_q1 = beta ** (eta_c + 1.0)
                else:
                    beta_q1 = 1.0 / (beta * (2.0 * beta - 1.0) + 1.0) ** (1.0 / (eta_c + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta_q1) * x1[i] + (1.0 - beta_q1) * x2[i])
                
                # Calculate beta_q2
                if rng.random() * 2 <= 1.0 / (beta + 1.0):
                    beta_q2 = beta ** (eta_c + 1.0)
                else:
                    beta_q2 = 1.0 / (beta * (2.0 * beta - 1.0) + 1.0) ** (1.0 / (eta_c + 1.0))
                
                child2[i] = 0.5 * ((1.0 + beta_q2) * x2[i] + (1.0 - beta_q2) * x1[i])
            
            # Clip to bounds
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        def polynomial_mutation(x, rng):
            """Polynomial Mutation"""
            child = x.copy()
            for i in range(dim):
                if rng.random() < mut_prob:
                    if rng.random() < 0.5:
                        delta1 = (child[i] - lo[i]) / range_[i]
                        mut_pow = 1.0 / (eta_m + 1.0)
                        xq = 1.0 + (2.0 * (1.0 - delta1)) ** mut_pow
                        if rng.random() * 2 <= 1.0 / xq:
                            deltaq = (xq ** (eta_m + 1.0) - 1.0) / 2.0
                        else:
                            deltaq = 1.0 - (2.0 * (1.0 - 1.0 / xq)) ** (1.0 / (eta_m + 1.0))
                        child[i] -= deltaq * range_[i]
                    else:
                        delta2 = (hi[i] - child[i]) / range_[i]
                        mut_pow = 1.0 / (eta_m + 1.0)
                        xq = 1.0 + (2.0 * (1.0 - delta2)) ** mut_pow
                        if rng.random() * 2 <= 1.0 / xq:
                            deltaq = (xq ** (eta_m + 1.0) - 1.0) / 2.0
                        else:
                            deltaq = 1.0 - (2.0 * (1.0 - 1.0 / xq)) ** (1.0 / (eta_m + 1.0))
                        child[i] += deltaq * range_[i]
            return np.clip(child, lo, hi)

        def tournament_select(individuals, ranks, distances, rng):
            """Binary Tournament Selection"""
            i1 = rng.integers(len(individuals))
            i2 = rng.integers(len(individuals))
            
            if ranks[i1] != ranks[i2]:
                return individuals[i1] if ranks[i1] < ranks[i2] else individuals[i2]
            else:
                return individuals[i1] if distances[i1] > distances[i2] else individuals[i2]

        while evaluations < budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            while len(offspring_X) < pop_size:
                p1 = tournament_select(X, rank, dist, rng)
                p2 = tournament_select(X, rank, dist, rng)
                c1, c2 = sbx_crossover(p1, p2, rng)
                c1 = polynomial_mutation(c1, rng)
                c2 = polynomial_mutation(c2, rng)
                offspring_X.append(c1)
                offspring_X.append(c2)
                evaluations += 2
                if evaluations > budget:
                    break
                    
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            
            # Combine parent and offspring
            X_r = np.vstack((X, offspring_X))
            F_r = np.vstack((F, offspring_F))
            
            # Non-dominated sort
            fronts, rank = fast_non_dominated_sort(F_r)
            
            # Select next generation
            X_new = []
            F_new = []
            dist = np.zeros(len(X_r))
            
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    X_new.extend([X_r[i] for i in front])
                    F_new.extend([F_r[i] for i in front])
                    # Calculate crowding distance for this front
                    front_F = F_r[front]
                    front_dist = crowding_distance(front_F, list(range(len(front))))
                    for idx, i in enumerate(front):
                        dist[i] = front_dist[idx]
                else:
                    # Need to select from this front based on crowding distance
                    front_F = F_r[front]
                    front_dist = crowding_distance(front_F, list(range(len(front))))
                    remaining = pop_size - len(X_new)
                    sorted_indices = np.argsort(-front_dist)
                    selected = [front[i] for i in sorted_indices[:remaining]]
                    X_new.extend([X_r[i] for i in selected])
                    F_new.extend([F_r[i] for i in selected])
                    for idx, i in enumerate(front):
                        if i in selected:
                            dist[i] = front_dist[idx]
                    break
                    
            X = np.array(X_new)
            F = np.array(F_new)
            
            # Recalculate ranks and distances for next selection round
            fronts, rank = fast_non_dominated_sort(F)
            dist = np.zeros(len(X))
            for front in fronts:
                if len(front) > 2:
                    front_F = F[front]
                    front_dist = crowding_distance(front_F, list(range(len(front))))
                    for idx, i in enumerate(front):
                        dist[i] = front_dist[idx]
                else:
                    for i in front:
                        dist[i] = np.inf

        return X
# EVOLVE-BLOCK-END
