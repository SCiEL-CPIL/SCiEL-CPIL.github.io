"""Rewritten program using Multi-Objective Particle Swarm Optimization (MOPSO)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOPSO inspired multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Parameters
        pop_size = 15
        w = 0.9  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        max_repo_size = 30
        noise_scale = 0.05

        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, n))
        V = np.zeros((pop_size, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Initialize Personal Bests and Repository
        Pbest = X.copy()
        Pbest_F = F.copy()

        def is_dominated(a, b):
            # Returns True if a dominates b
            return np.all(a <= b) and np.any(a < b)

        # Initialize Repository
        repo = []
        for i in range(pop_size):
            # Check if X[i] is dominated by any in repo
            is_dom = False
            for rx, rf in repo:
                if is_dominated(rf, F[i]):
                    is_dom = True
                    break
            if not is_dom:
                # Remove points from repo dominated by X[i]
                new_repo = []
                for rx, rf in repo:
                    if not is_dominated(F[i], rf):
                        new_repo.append((rx, rf))
                new_repo.append((X[i].copy(), F[i].copy()))
                repo = new_repo

        def crowding_distance(repo_list):
            """Calculate crowding distance for points in repo_list (list of (x, f))."""
            if len(repo_list) <= 2:
                return [float('inf')] * len(repo_list)
            
            # Extract F
            Fs = np.array([f for _, f in repo_list])
            m = len(Fs)
            cd = [0.0] * m
            
            for d in range(self.n_obj):
                obj = Fs[:, d]
                order = np.argsort(obj)
                cd_sorted = [0.0] * m
                cd_sorted[order[0]] = float('inf')
                cd_sorted[order[-1]] = float('inf')
                obj_range = obj[order[-1]] - obj[order[0]]
                if obj_range > 1e-12:
                    for i in range(1, m - 1):
                        cd_sorted[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / obj_range
                else:
                    for i in range(1, m - 1):
                        cd_sorted[order[i]] += float('inf')
                for i in range(m):
                    cd[i] += cd_sorted[i]
            return cd

        def select_leader(repo_list):
            """Select a leader from repo based on crowding distance."""
            if len(repo_list) == 0:
                return None
            if len(repo_list) == 1:
                return repo_list[0][0]
            
            cd = crowding_distance(repo_list)
            # Handle inf in weights
            finite_cds = [c for c in cd if np.isfinite(c)]
            max_cd = max(finite_cds) if finite_cds else 0
            cd_clean = [max_cd + 1 if not np.isfinite(c) else c for c in cd]
            weights = np.array(cd_clean)
            weights = weights / np.sum(weights)
            idx = rng.choice(len(repo_list), p=weights)
            return repo_list[idx][0]

        # Main Loop
        while used < budget:
            # Update each particle
            for i in range(pop_size):
                if used >= budget:
                    break
                
                # Select Leader
                leader = select_leader(repo)
                if leader is None:
                    leader = X[i] # Fallback
                
                # Update Velocity with noise
                r1 = rng.random()
                r2 = rng.random()
                V[i] = w * V[i] + c1 * r1 * (Pbest[i] - X[i]) + c2 * r2 * (leader - X[i])
                V[i] += noise_scale * (hi - lo) * rng.standard_normal(n)
                
                # Update Position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                new_F = problem(X[i])
                used += 1
                
                # Update Pbest
                # If new dominates Pbest, update
                # If Pbest dominates new, keep
                # If non-dominated, keep new (to allow exploration)
                if is_dominated(new_F, Pbest_F[i]):
                    Pbest[i] = X[i].copy()
                    Pbest_F[i] = new_F.copy()
                elif not is_dominated(Pbest_F[i], new_F):
                    # Non-dominated
                    Pbest[i] = X[i].copy()
                    Pbest_F[i] = new_F.copy()
                
                # Update Repository
                # Check if new_F is dominated by any in repo
                is_dom_by_repo = False
                for rx, rf in repo:
                    if is_dominated(rf, new_F):
                        is_dom_by_repo = True
                        break
                
                if not is_dom_by_repo:
                    # Remove points from repo dominated by new_F
                    new_repo = []
                    for rx, rf in repo:
                        if not is_dominated(new_F, rf):
                            new_repo.append((rx, rf))
                    new_repo.append((X[i].copy(), new_F.copy()))
                    repo = new_repo
                    
                    # Enforce max size
                    if len(repo) > max_repo_size:
                        cd = crowding_distance(repo)
                        # Find index with min crowding distance
                        min_idx = np.argmin(cd)
                        repo.pop(min_idx)

        return X
# EVOLVE-BLOCK-END
