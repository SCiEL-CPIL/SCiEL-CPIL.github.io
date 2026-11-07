import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Population size
        # For 4D, 20 is a good balance for 200 budget.
        self.pop_size = 20
        self._start_time = None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        
        rng = np.random.default_rng(0)
        
        # Initialization
        X = lo + (hi - lo) * rng.random((self.pop_size, d))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        while used + self.pop_size <= self.budget:
            # Calculate ranks and CD for current population X
            ranks, cd = self._calculate_rank_cd(F)
            
            # Create offspring
            Q = np.zeros((self.pop_size, d))
            FQ = np.zeros((self.pop_size, self.n_obj))
            
            for i in range(self.pop_size):
                p1 = self._tournament(ranks, cd, rng)
                p2 = self._tournament(ranks, cd, rng)
                
                x1 = X[p1]
                x2 = X[p2]
                
                # Crossover: BLX-Alpha
                alpha = 0.5
                low = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
                high = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
                child = rng.uniform(low, high, size=d)
                child = np.clip(child, lo, hi)
                
                # Mutation
                # Gaussian mutation
                for k in range(d):
                    if rng.random() < 1.0/d:
                        child[k] += 0.1 * (hi[k] - lo[k]) * rng.standard_normal()
                        child[k] = np.clip(child[k], lo[k], hi[k])
                
                Q[i] = child
                FQ[i] = problem(child)
            
            used += self.pop_size
            
            # Combine and select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            X, F = self._nsga2_select(R_X, R_F)
            
        # Tail: use remaining budget
        while used < self.budget:
            x = lo + (hi - lo) * rng.random(d)
            fx = problem(x)
            used += 1
            # Simple replacement: replace worst sum
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            # Only replace if not dominated? 
            # To be safe and improve front, replace if new point is non-dominated in X U {x}
            # Check dominance
            dominated = False
            for j in range(len(F)):
                if np.all(F[j] <= fx) and np.any(F[j] < fx):
                    dominated = True
                    break
            
            if not dominated:
                # Remove any points dominated by fx
                new_X = []
                new_F = []
                for j in range(len(F)):
                    if not (np.all(fx <= F[j]) and np.any(fx < F[j])):
                        new_X.append(X[j])
                        new_F.append(F[j])
                # Add fx
                new_X.append(x)
                new_F.append(fx)
                # If we have more than pop_size, trim?
                # No, we are in tail, we just have pop_size points in X.
                # If we added one, we have pop_size + 1.
                # We should pick the best pop_size.
                # But this is O(N^2) every step.
                # Given tail is small, it's fine.
                if len(new_X) > self.pop_size:
                    # Select best pop_size
                    R_X = np.array(new_X)
                    R_F = np.array(new_F)
                    X, F = self._nsga2_select(R_X, R_F)
                else:
                    X = np.array(new_X)
                    F = np.array(new_F)
            else:
                # If dominated, just discard? Or replace worst?
                # Discarding wastes the eval.
                # Replacing worst sum is better than nothing.
                X[worst_idx] = x
                F[worst_idx] = fx

        return X

    def _calculate_rank_cd(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)
        
        # Non-dominated sorting
        remaining = set(range(n))
        current_rank = 0
        
        while remaining:
            front_indices = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
                    remaining.remove(i)
            
            if not front_indices:
                break
                
            for i in front_indices:
                ranks[i] = current_rank
            
            if len(front_indices) > 1:
                for m in range(self.n_obj):
                    order = np.argsort(F[front_indices, m])
                    sorted_indices = [front_indices[k] for k in order]
                    
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    
                    span = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if span > 0:
                        for k in range(1, len(sorted_indices)-1):
                            idx_curr = sorted_indices[k]
                            idx_prev = sorted_indices[k-1]
                            idx_next = sorted_indices[k+1]
                            cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / span
            else:
                for i in front_indices:
                    cd[i] = np.inf
            
            current_rank += 1
            
        return ranks, cd

    def _tournament(self, ranks, cd, rng):
        i1 = rng.integers(len(ranks))
        i2 = rng.integers(len(ranks))
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1

    def _nsga2_select(self, R_X, R_F):
        ranks, cd = self._calculate_rank_cd(R_F)
        
        # Sort by rank asc, then cd desc
        # np.lexsort keys: last is primary.
        # Primary: ranks (asc)
        # Secondary: -cd (asc) -> cd desc
        sorted_indices = np.lexsort((-cd, ranks))
        
        selected_idx = sorted_indices[:self.pop_size]
        
        return R_X[selected_idx], R_F[selected_idx]
