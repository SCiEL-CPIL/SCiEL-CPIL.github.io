"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(20, self.budget // 5))
        
        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        mu = pop_size  # parent population size
        eta = pop_size  # offspring population size
        
        while used + eta <= self.budget and time.time() < deadline:
            # Create offspring
            children_X = []
            children_F = []
            
            for i in range(eta):
                if time.time() > deadline:
                    break
                
                # Tournament selection (binary tournament)
                idx1 = rng.integers(mu)
                idx2 = rng.integers(mu)
                # Select parent with lower rank (need to compute ranks first)
                # For simplicity, use non-dominated front selection
                parent_idx = self._tournament_select(F, rng, mu)
                
                # Mutation: Gaussian perturbation
                sigma = 0.15 * (hi - lo)
                child = X[parent_idx] + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)
            
            if not children_X:
                break
                
            children_X = np.array(children_X)
            children_F = np.array(children_F)
            
            # Combine parents and offspring
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_F)
            
            # Build next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= mu:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to fill remaining slots by crowding distance
                    remaining = mu - len(new_X)
                    # Sort front by crowding distance
                    front_F = combined_F[front]
                    cd = self._crowding_distance(front_F)
                    sorted_indices = np.argsort(-cd)
                    for idx in sorted_indices[:remaining]:
                        orig_idx = front[idx]
                        new_X.append(combined_X[orig_idx])
                        new_F.append(combined_F[orig_idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _tournament_select(self, F, rng, size):
        """Binary tournament selection based on dominance."""
        idx1 = rng.integers(size)
        idx2 = rng.integers(size)
        
        f1 = F[idx1]
        f2 = F[idx2]
        
        # Check dominance
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return idx1
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return idx2
        else:
            # Incomparable, use random
            return idx1 if rng.random() < 0.5 else idx2

    def _non_dominated_sort(self, F):
        """Non-dominated sorting. Returns list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]  # Set of solutions dominated by i
        n_dominates = [0] * n  # Number of solutions that dominate i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                fi = F[i]
                fj = F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dominates[i] == 0]
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of solutions in the same front."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = F.shape[1]  # number of objectives
        
        for k in range(m):
            # Sort by objective k
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F[order[-1], k] - F[order[0], k]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / f_range
        
        return cd
# EVOLVE-BLOCK-END
