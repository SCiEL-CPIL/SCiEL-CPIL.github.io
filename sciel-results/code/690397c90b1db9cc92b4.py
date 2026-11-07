import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # 1. Initial Sampling: Latin Hypercube for better coverage
        n_init = min(40, self.budget)
        X_init = np.empty((n_init, dim))
        for i in range(dim):
            X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
        idx = np.random.permutation(n_init)
        X_init = X_init[idx]
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Track all evaluated points for potential final return (though return is ignored, 
        # the contract says "Every returned objective vector is recorded", implying we just 
        # need to call problem. The return value of __call__ is ignored. 
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This usually means the evaluator intercepts `problem` calls. 
        # So we just need to make sure we call `problem` correctly.
        # The return value of `__call__` is ignored, so we can return anything.
        
        # 2. Setup Population for Evolution
        pop_size = min(30, self.budget - used)
        if pop_size < 5:
            return X_init
            
        # Initialize population from initial samples
        # Use non-dominated points if available, otherwise random
        nd_idx = self._get_non_dominated(F_init)
        
        X_pop = np.empty((pop_size, dim))
        F_pop = np.empty((pop_size, n_obj))
        
        if len(nd_idx) >= pop_size:
            X_pop = X_init[nd_idx[:pop_size]].copy()
            F_pop = F_init[nd_idx[:pop_size]].copy()
        else:
            X_pop[:len(nd_idx)] = X_init[nd_idx]
            F_pop[:len(nd_idx)] = F_init[nd_idx]
            remaining = pop_size - len(nd_idx)
            all_idx = np.arange(len(X_init))
            nd_set = set(nd_idx.tolist())
            available = [i for i in all_idx if i not in nd_set]
            if len(available) >= remaining:
                fill_idx = np.random.choice(available, remaining, replace=False)
            else:
                fill_idx = np.random.choice(len(X_init), remaining, replace=True)
            X_pop[len(nd_idx):] = X_init[fill_idx]
            F_pop[len(nd_idx):] = F_init[fill_idx]
            
        # 3. Evolutionary Loop (DE-like with NSGA-II selection)
        F_mut = 0.5
        CR = 0.7
        
        # Track best non-dominated solutions seen so far
        archive_X = [X_pop[i].copy() for i in range(pop_size)]
        archive_F = [F_pop[i].copy() for i in range(pop_size)]
        
        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def _crowding_distance(F_subset):
            """Compute crowding distance for points in F_subset."""
            n = len(F_subset)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for k in range(F_subset.shape[1]):
                order = np.argsort(F_subset[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    denom = F_subset[order[-1], k] - F_subset[order[0], k]
                    if denom > 0:
                        cd[order[i]] += (F_subset[order[i+1], k] - F_subset[order[i-1], k]) / denom
            return cd
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate one offspring at a time to save budget and allow frequent checks
            i = np.random.randint(pop_size)
            
            # Select three distinct random indices r1, r2, r3 != i
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                # If pop size is small, just pick with replacement
                r1, r2, r3 = np.random.choice(pop_size, size=3, replace=True)
            else:
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
            
            # Mutation (DE/rand/1)
            v = X_pop[r1] + F_mut * (X_pop[r2] - X_pop[r3])
            
            # Crossover (Binomial)
            j_rand = np.random.randint(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X_pop[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection: tournament-style with dominance and crowding
            if _dominates(f_trial, F_pop[i]):
                X_pop[i] = trial
                F_pop[i] = f_trial
            elif not _dominates(F_pop[i], f_trial):
                # Non-dominated: use crowding distance
                # Add trial to population temporarily and compute crowding
                temp_F = np.vstack([F_pop, f_trial.reshape(1, -1)])
                cd = _crowding_distance(temp_F)
                cd_i = cd[i]
                cd_trial = cd[pop_size]
                
                if cd_trial > cd_i:
                    X_pop[i] = trial
                    F_pop[i] = f_trial
                elif np.random.random() < 0.2:
                    X_pop[i] = trial
                    F_pop[i] = f_trial
            
            # Update archive with non-dominated solutions
            is_dom_in_archive = False
            for k in range(len(archive_F)):
                if _dominates(f_trial, archive_F[k]):
                    # Remove dominated archive members
                    archive_X.pop(k)
                    archive_F.pop(k)
                    k -= 1
                elif _dominates(archive_F[k], f_trial):
                    is_dom_in_archive = True
                    break
            
            if not is_dom_in_archive:
                archive_X.append(trial.copy())
                archive_F.append(f_trial.copy())
        
        # Final: return the archive (best non-dominated solutions)
        return np.array(archive_X)

    def _get_non_dominated(self, F):
        """
        Returns indices of non-dominated points in F.
        F is (N, n_obj). Minimization assumed.
        """
        N = len(F)
        if N == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(N, dtype=bool)
        
        for i in range(N):
            if is_dominated[i]:
                continue
            for j in range(N):
                if i == j or is_dominated[j]:
                    continue
                # Check if i dominates j
                # i dominates j if i <= j and i < j in at least one dimension
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
        
        return np.where(~is_dominated)[0]
