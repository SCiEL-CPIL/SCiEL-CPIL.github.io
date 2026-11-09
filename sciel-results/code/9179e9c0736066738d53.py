import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOPSO (Multi-Objective Particle Swarm Optimization)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if vector a dominates vector b."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if self._dominates(F[i], F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[indices, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F[indices[order[-1]], m] - F[indices[order[0]], m]
            if f_range > 0:
                for k in range(1, n - 1):
                    dist[order[k]] += (F[indices[order[k + 1]], m] - F[indices[order[k - 1]], m]) / f_range
        return dist

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Swarm size
        swarm_size = 50
        
        # Initialize positions uniformly
        X = rng.random((swarm_size, self.dim)) * span + lo
        V = np.zeros((swarm_size, self.dim))
        
        # Evaluate initial swarm
        F = np.array([problem(x) for x in X])
        used = swarm_size
        
        # Initialize personal bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Initialize global archive
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any in archive
            dominated = False
            for af in archive_F:
                if self._dominates(af, f):
                    dominated = True
                    break
            if dominated:
                return
            
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not self._dominates(f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = np.array(new_archive_F)
            
            # Trim archive if too large
            if len(archive_F) > 100:
                non_dom_idx = self._non_dominated_indices(archive_F)
                if len(non_dom_idx) > 100:
                    cd = self._crowding_distance(archive_F, non_dom_idx)
                    order = np.argsort(cd)[::-1][:100]
                    archive_X = [archive_X[i] for i in non_dom_idx[order]]
                    archive_F = archive_F[non_dom_idx[order]]

        # Add initial points to archive
        for i in range(swarm_size):
            update_archive(X[i], F[i])

        # PSO Parameters
        w = 0.9  # Inertia weight
        c1 = 1.4 # Cognitive coefficient
        c2 = 1.4 # Social coefficient
        max_v = 0.2 * span  # Max velocity

        while used < self.budget:
            # Select leader for the swarm
            # If archive is empty, use random points
            if len(archive_F) == 0:
                leader_idx = rng.integers(swarm_size)
                leader = X[leader_idx]
            else:
                # Select leader using crowding distance roulette
                cd = self._crowding_distance(archive_F, np.arange(len(archive_F)))
                cd_sum = np.sum(cd)
                if cd_sum == 0:
                    leader_idx = rng.integers(len(archive_F))
                else:
                    probs = cd / cd_sum
                    leader_idx = rng.choice(len(archive_F), p=probs)
                leader = archive_X[leader_idx]
            
            # Update velocities and positions for all particles
            for i in range(swarm_size):
                if used >= self.budget:
                    break
                
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                # Update velocity
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (leader - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -max_v, max_v)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Handle boundaries: reflect velocity and clip position
                for d in range(self.dim):
                    if X[i, d] < lo[d]:
                        X[i, d] = lo[d]
                        V[i, d] = -V[i, d]
                    elif X[i, d] > hi[d]:
                        X[i, d] = hi[d]
                        V[i, d] = -V[i, d]
                
                # Evaluate
                f_new = problem(X[i])
                used += 1
                
                # Update personal best
                if self._dominates(f_new, Pbest_F[i]):
                    Pbest_F[i] = f_new
                    Pbest_X[i] = X[i].copy()
                
                # Update archive
                update_archive(X[i], f_new)

        return archive_X
