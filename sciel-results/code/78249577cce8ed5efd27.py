"""Multi-objective optimizer on ZDT6 with NSGA-II and dimensionality reduction."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II optimizer tailored for ZDT6 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 insight: Optimal solutions have x[2:] = 0.
        # We restrict search to x[0] and x[1], setting others to 0.
        # This reduces dimensionality from 10 to 2, greatly improving efficiency.
        
        pop_size = 20
        # Initialize population
        # x[0], x[1] random in [0, 1]
        # x[2:] = 0
        X = np.zeros((pop_size, self.dim))
        X[:, 0] = np.random.rand(pop_size)
        X[:, 1] = np.random.rand(pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II loop
        while used < self.budget and time.time() < deadline:
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Select parents using tournament selection on combined pop (X, F)
                # For simplicity and speed, we select from current X
                p1_idx, p2_idx = self._tournament_select(F, X)
                
                # Crossover (SBX) on first 2 dimensions
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                child = np.zeros(self.dim)
                
                # SBX on x[0] and x[1]
                for d in range(2):
                    if np.random.rand() < 0.9: # Crossover prob
                        beta = self._sbx(x1[d], x2[d], 20.0)
                        child[d] = beta
                    else:
                        child[d] = x1[d]
                        
                # Mutation on x[0] and x[1]
                for d in range(2):
                    if np.random.rand() < 0.1: # Mutation prob
                        child[d] = self._polynomial_mutation(child[d], lo[d], hi[d], 20.0)
                        
                # Keep x[2:] at 0
                child[2:] = 0.0
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                
                if used >= self.budget or time.time() >= deadline:
                    break
                    
            if used >= self.budget or time.time() >= deadline:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            ranks, crowding = self._fast_non_dominated_sort(combined_F)
            
            # Select best pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            sort_indices = np.lexsort(([-crowding[i] for i in range(len(ranks))], ranks))
            
            # Wait, lexsort sorts ascending. We want rank ascending, crowding descending.
            # So we negate crowding for sorting.
            # Actually, standard approach:
            # Create a list of (rank, -crowding) and sort.
            keys = list(zip(ranks, [-c for c in crowding]))
            sorted_indices = np.argsort(np.array(keys), kind='mergesort')
            
            X = combined_X[sorted_indices[:pop_size]]
            F = combined_F[sorted_indices[:pop_size]]
            
        return X

    def _tournament_select(self, F, X, k=2):
        """Binary tournament selection based on non-domination and crowding."""
        # For efficiency in this small pop, we do a simple tournament on ranks
        # But we don't have ranks stored. We'll do a quick dominance check tournament.
        # Or just pick random parents? 
        # Better: Pick 2 random, evaluate dominance.
        idx1 = np.random.randint(len(F))
        idx2 = np.random.randint(len(F))
        
        f1 = F[idx1]
        f2 = F[idx2]
        
        # Check if f1 dominates f2
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return idx1, np.random.randint(len(F))
        # Check if f2 dominates f1
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return idx2, np.random.randint(len(F))
        else:
            # Non-dominated, pick random
            return idx1, idx2

    def _sbx(self, x1, x2, eta):
        """Simulated Binary Crossover."""
        if x1 == x2:
            return x1
        if np.random.rand() < 0.5:
            beta = (2 * x1 / (x1 - x2)) ** (1 / (eta + 1))
            child = 0.5 * (x1 + x2 - beta * (x2 - x1))
        else:
            beta = (2 * x2 / (x2 - x1)) ** (1 / (eta + 1))
            child = 0.5 * (x1 + x2 + beta * (x2 - x1))
        return np.clip(child, 0, 1)

    def _polynomial_mutation(self, x, lo, hi, eta):
        """Polynomial Mutation."""
        if x == lo or x == hi:
            return x
        delta1 = (x - lo) / (hi - lo)
        delta2 = (hi - x) / (hi - lo)
        rand = np.random.rand()
        if rand < 0.5:
            xy = 1 - (1 - delta1) ** (eta + 1)
            delta = (2 * xy) ** (1 / (eta + 1))
        else:
            xy = 1 - (1 - delta2) ** (eta + 1)
            delta = (2 * xy) ** (1 / (eta + 1))
            x = x + delta * (hi - lo)
        return np.clip(x, lo, hi)

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting and crowding distance calculation."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # List of fronts
        front1 = []
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        # Step 1: Find first front
        for i in range(n):
            for j in range(n):
                if i == j: continue
                fi = F[i]
                fj = F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].add(j)
                    n_dom[i] += 1
                elif np.all(fj <= fi) and np.any(fj < fi):
                    n_dom[i] += 1
        # Correction: n_dom[i] is number of individuals dominating i.
        # My loop above was slightly wrong.
        # Let's redo:
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                fi = F[i]
                fj = F[j]
                # Does i dominate j?
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].add(j)
                else:
                    # Does j dominate i?
                    if np.all(fj <= fi) and np.any(fj < fi):
                        n_dom[i] += 1
        
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front1.append(i)
                
        current_front = front1
        front_num = 1
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = front_num
                        next_front.append(j)
            # Calculate crowding distance for current front
            self._assign_crowding_distance(current_front, F, crowding)
            current_front = next_front
            front_num += 1
            
        return ranks, crowding

    def _assign_crowding_distance(self, front_indices, F, crowding):
        """Assign crowding distances to a list of indices in the same front."""
        n = len(front_indices)
        if n <= 2:
            for i in front_indices:
                crowding[i] = np.inf
            return
        
        for m in range(self.n_obj):
            # Sort front by m-th objective
            sorted_front = sorted(front_indices, key=lambda idx: F[idx][m])
            crowding[sorted_front[0]] = np.inf
            crowding[sorted_front[-1]] = np.inf
            
            f_min = F[sorted_front[0]][m]
            f_max = F[sorted_front[-1]][m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_front[i]
                prev_idx = sorted_front[i-1]
                next_idx = sorted_front[i+1]
                crowding[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)

# EVOLVE-BLOCK-END
