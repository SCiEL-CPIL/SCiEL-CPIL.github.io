"""New optimizer: Multi-Objective Particle Swarm Optimization (MOPSO)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Particle Swarm Optimization (MOPSO) for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Reference point for normalization/scoring context
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Swarm parameters
        # Budget 200. Swarm size 20 allows 10 generations.
        swarm_size = 20
        # Ensure we don't exceed budget in init
        if swarm_size >= budget:
            swarm_size = max(1, budget - 1) # Leave at least 1 for loop? Or just use all?
            # If budget is small, we can't do much.
        
        # Initialize swarm positions
        X = lo + span * rng.random((swarm_size, dim))
        # Initialize velocities
        V = 0.1 * span * rng.random((swarm_size, dim))
        
        # Personal bests
        Pbest_X = X.copy()
        Pbest_F = np.array([problem(x) for x in X], dtype=float)
        
        used = swarm_size
        
        # Archive initialization
        archive_X = list(X)
        archive_F = list(Pbest_F)
        
        # Prune archive initially
        archive_X, archive_F = self._prune_archive(archive_X, archive_F, ref)

        # PSO Parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 1.49445 # Standard PSO constants
        c2 = 1.49445
        max_generations = max(1, (budget - used) // swarm_size)

        for gen in range(max_generations):
            if used >= budget:
                break

            # Select global best from archive
            # Use crowding distance to pick a diverse representative
            gbest_X, gbest_F = self._select_global_best(archive_X, archive_F, ref, rng)
            
            # Linearly decrease inertia
            w = w_start - (w_start - w_end) * (gen / max_generations)
            
            # Synchronous Update
            new_V = np.zeros_like(V)
            new_X = np.zeros_like(X)
            
            # We need to evaluate all particles in this generation
            # But we must respect the budget
            num_evals_this_gen = min(swarm_size, budget - used)
            
            for i in range(num_evals_this_gen):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Velocity update
                new_V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
                
                # Position update
                new_X[i] = X[i] + new_V[i]
                
                # Clip to bounds
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Evaluate
                F = problem(new_X[i])
                used += 1
                
                # Update personal best
                if self._dominates(F, Pbest_F[i]):
                    Pbest_X[i] = new_X[i].copy()
                    Pbest_F[i] = F.copy()
                
                # Update archive
                archive_X, archive_F = self._update_archive(archive_X, archive_F, new_X[i], F, ref)

            # Update state for next generation
            # Note: Only the first 'num_evals_this_gen' particles were updated.
            # If num_evals < swarm_size, the rest keep their old X/V? 
            # In standard PSO, all update. If budget runs out mid-gen, we stop.
            # We should only update X/V for the ones we actually moved.
            V[:num_evals_this_gen] = new_V[:num_evals_this_gen]
            X[:num_evals_this_gen] = new_X[:num_evals_this_gen]

        # Return the best point (ignored by evaluator, but required)
        return Pbest_X[0]

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _prune_archive(self, archive_X, archive_F, ref):
        """Remove dominated points from archive."""
        if len(archive_F) == 0:
            return archive_X, archive_F
        
        F_arr = np.array(archive_F)
        X_arr = np.array(archive_X)
        
        n = len(F_arr)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if self._dominates(F_arr[i], F_arr[j]):
                    is_dominated[j] = True
        
        keep_idx = np.where(~is_dominated)[0]
        
        # Limit archive size
        max_archive_size = 100
        if len(keep_idx) > max_archive_size:
            # Use crowding distance to prune
            F_sub = F_arr[keep_idx]
            cd = self._crowding_distance(F_sub)
            # Keep points with highest crowding distance
            keep_local = np.argsort(-cd)[:max_archive_size]
            keep_idx = keep_idx[keep_local]
            
        return [X_arr[i] for i in keep_idx], [F_arr[i] for i in keep_idx]

    def _crowding_distance(self, F_arr):
        """Compute crowding distance for a set of non-dominated points."""
        n = len(F_arr)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        n_obj = F_arr.shape[1]
        for m in range(n_obj):
            vals = F_arr[:, m]
            mn, mx = vals.min(), vals.max()
            if mx - mn < 1e-15:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for k in range(1, n - 1):
                cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
        return cd

    def _select_global_best(self, archive_X, archive_F, ref, rng):
        """Select a global best from the archive using crowding distance."""
        if not archive_F:
            return np.zeros(len(ref)), np.zeros(len(ref))
        
        F_arr = np.array(archive_F)
        X_arr = np.array(archive_X)
        
        # If only one point, return it
        if len(F_arr) == 1:
            return X_arr[0], F_arr[0]
        
        # Compute crowding distance
        cd = self._crowding_distance(F_arr)
        
        # Select point with max crowding distance (most diverse)
        # To avoid always picking the same extreme, we can pick randomly from top 10%
        top_k = max(1, len(F_arr) // 10)
        top_indices = np.argsort(-cd)[:top_k]
        chosen_idx = top_indices[rng.integers(len(top_indices))]
        
        return X_arr[chosen_idx], F_arr[chosen_idx]

    def _update_archive(self, archive_X, archive_F, new_x, new_f, ref):
        """Update archive with a new point."""
        # Check if new point is dominated by any in archive
        for f in archive_F:
            if self._dominates(f, new_f):
                return archive_X, archive_F
        
        # Add new point
        archive_X.append(new_x)
        archive_F.append(new_f)
        
        # Remove points dominated by new point
        new_archive_X = []
        new_archive_F = []
        for i, f in enumerate(archive_F):
            if not self._dominates(new_f, f):
                new_archive_X.append(archive_X[i])
                new_archive_F.append(f)
            # If new_f dominates f, we drop f. 
            # Note: if they are equal, we keep one (the old one, effectively dropping new one? 
            # No, we added new one first. So we drop the old one if new dominates old.
            # If old dominates new, we already returned.
            # If neither dominates, keep both.
            
        # Prune if too large
        if len(new_archive_F) > 100:
            new_archive_X, new_archive_F = self._prune_archive(new_archive_X, new_archive_F, ref)
            
        return new_archive_X, new_archive_F
# EVOLVE-BLOCK-END
