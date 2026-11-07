import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Population size
        self.pop_size = 20

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        pop_size = self.pop_size

        # LHS Initialization for better spread
        X = np.zeros((pop_size, d))
        for i in range(d):
            edges = np.linspace(lo[i], hi[i], pop_size + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            np.random.shuffle(centers)
            X[:, i] = centers

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_X = X.copy()
        archive_F = F.copy()

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(new_x, new_f, archive_X, archive_F):
            """Add new point to archive if non-dominated, remove dominated archive points."""
            if is_dominated(new_f, archive_F):
                return archive_X, archive_F

            keep = []
            for i, af in enumerate(archive_F):
                if not (np.all(new_f <= af) and np.any(new_f < af)):
                    keep.append(i)

            if keep:
                new_archive_X = np.vstack([archive_X[keep], new_x])
                new_archive_F = np.vstack([archive_F[keep], new_f])
            else:
                new_archive_X = new_x.reshape(1, -1)
                new_archive_F = new_f.reshape(1, -1)
            return new_archive_X, new_archive_F

        # DE Parameters
        Fp = 0.8  # Mutation factor
        Cr = 0.8  # Crossover rate

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Calculate ranks and CD for current population
            ranks, cd = self._calculate_rank_cd(F)

            # Generate offspring using DE/rand/1
            a = np.random.randint(pop_size)
            b = np.random.randint(pop_size)
            c = np.random.randint(pop_size)
            
            # Ensure distinct indices if possible (though not strictly required for DE)
            if a == b or a == c or b == c:
                # Retry to get distinct
                while a == b or a == c or b == c:
                    a = np.random.randint(pop_size)
                    b = np.random.randint(pop_size)
                    c = np.random.randint(pop_size)
                    # Safety break if we loop too much (unlikely)
                    if np.random.randint(10) == 0:
                        break

            mutant = X[a] + Fp * (X[b] - X[c])
            mutant = np.clip(mutant, lo, hi)

            # Binary Crossover
            trial = X[a].copy()
            jrand = np.random.randint(d)
            for j in range(d):
                if np.random.random() < Cr or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = X[a][j]
            
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update Archive
            archive_X, archive_F = update_archive(trial, f_trial, archive_X, archive_F)

            # Selection: Replace parent if trial is better in some sense
            # Standard DE selection: if trial dominates parent, or has better fitness
            # For multi-objective, we use dominance check.
            # If trial dominates parent: replace
            # If parent dominates trial: keep parent
            # If non-dominated: use ranking/crowding or random
            # Simple strategy: Replace if trial is non-dominated by parent AND 
            # (trial dominates parent OR random probability)
            
            # Check dominance
            trial_dom_parent = np.all(f_trial <= F[a]) and np.any(f_trial < F[a])
            parent_dom_trial = np.all(F[a] <= f_trial) and np.any(F[a] < f_trial)

            if trial_dom_parent:
                X[a] = trial
                F[a] = f_trial
            elif not parent_dom_trial:
                # Non-dominated relation
                # Use a simple heuristic: replace with some probability to maintain diversity
                if np.random.random() < 0.5:
                    X[a] = trial
                    F[a] = f_trial
            # Else: Parent dominates trial, keep parent

        # Final cleanup: ensure we return something valid
        if len(archive_F) > 0:
            return archive_X
        else:
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
        sorted_indices = np.lexsort((-cd, ranks))
        
        selected_idx = sorted_indices[:self.pop_size]
        
        return R_X[selected_idx], R_F[selected_idx]
