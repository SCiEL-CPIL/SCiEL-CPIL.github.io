# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II approach.
    Uses Pareto dominance and crowding distance for selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_eval = 0

        # Helper to evaluate and count
        def eval_point(x):
            nonlocal n_eval
            x = np.clip(x, lo, hi)
            f = problem(x)
            n_eval += 1
            return x, f

        # Population size
        # Budget 200. N=20 allows 1 init + 9 generations.
        N = 20
        
        # Initialize population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        for i in range(N):
            x = lo + (hi - lo) * rng.random(self.dim)
            x, f = eval_point(x)
            X[i] = x
            F[i] = f
        
        # NSGA-II Parameters
        F_de = 0.8 # Scaling factor for DE mutation
        CR = 1.0   # We use DE mutation directly, so CR is implicit 1.0 for the mutation step
        
        # Main Loop
        while n_eval < self.budget:
            # Generate offspring Q
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            # Check if we have enough budget for a full generation
            if n_eval + N > self.budget:
                # Evaluate only as many as we can afford
                n_off = self.budget - n_eval
                # If n_off is 0, break
                if n_off == 0:
                    break
            else:
                n_off = N
            
            for i in range(n_off):
                # Select parents from X
                # DE/1/bin
                p1_idx = rng.integers(N)
                p2_idx = rng.integers(N)
                p3_idx = rng.integers(N)
                
                # Ensure distinct
                while p2_idx == p1_idx:
                    p2_idx = rng.integers(N)
                while p3_idx == p1_idx or p3_idx == p2_idx:
                    p3_idx = rng.integers(N)
                    
                v = X[p1_idx] + F_de * (X[p2_idx] - X[p3_idx])
                # Add small noise to escape local optima (DTLZ4 has sharp peak)
                noise_scale = 0.05 * (hi - lo)
                v = v + noise_scale * rng.standard_normal(self.dim)
                v = np.clip(v, lo, hi)
                
                v, f_v = eval_point(v)
                X_off[i] = v
                F_off[i] = f_v
            
            # Combine P and Q
            X_comb = np.vstack([X, X_off[:n_off]])
            F_comb = np.vstack([F, F_off[:n_off]])
            N_comb = len(X_comb)
            
            # Non-dominated sorting
            ranks, fronts = self.fast_non_dominated_sort(F_comb)
            
            # Select next generation P
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            
            idx = 0
            for front in fronts:
                if idx + len(front) <= N:
                    # Add all individuals in this front
                    for j in front:
                        X_new[idx] = X_comb[j]
                        F_new[idx] = F_comb[j]
                        idx += 1
                else:
                    # Need to select only some individuals from this front
                    k = N - idx
                    # Calculate crowding distance for this front
                    cd = self.crowding_distance(F_comb[front])
                    # Sort indices in this front by crowding distance (descending)
                    sorted_indices = np.argsort(-cd)
                    for j in range(k):
                        orig_idx = front[sorted_indices[j]]
                        X_new[idx] = X_comb[orig_idx]
                        F_new[idx] = F_comb[orig_idx]
                        idx += 1
                    break
            
            X = X_new
            F = F_new
            
            # If we didn't evaluate a full generation, we might be out of budget
            if n_eval >= self.budget:
                break

        return X

    def fast_non_dominated_sort(self, F):
        """
        Performs fast non-dominated sorting.
        Returns ranks array and list of fronts (each front is a list of indices).
        """
        n = len(F)
        S = [[] for _ in range(n)]  # Dominates
        n_dom = [0] * n             # Number of dominators
        rank = [0] * n              # Rank (front index)
        
        # O(N^2) dominance check
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = []
        
        for i in range(n):
            if n_dom[i] == 0:
                rank[i] = 0
                current_front.append(i)
                
        if current_front:
            fronts.append(current_front)
        
        k = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            if current_front:
                fronts.append(current_front)
            k += 1
            
        return rank, fronts

    def crowding_distance(self, F):
        """
        Calculates crowding distance for a set of points.
        F is a 2D array of shape (M, n_obj).
        Returns an array of length M.
        """
        M = len(F)
        if M <= 2:
            return np.full(M, np.inf)
            
        cd = np.zeros(M)
        
        for m in range(self.n_obj):
            # Sort indices by objective m
            sorted_indices = np.argsort(F[:, m])
            
            # Assign infinite distance to extremes
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            
            # Normalize objective range
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, M - 1):
                idx = sorted_indices[i]
                cd[idx] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / (f_max - f_min)
                
        return cd
# EVOLVE-BLOCK-END
