"""MOPSO based multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Particle Swarm Optimization (MOPSO)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Swarm parameters
        swarm_size = 40
        archive_size = 100
        w = 0.7  # Inertia weight
        c1 = 1.5 # Cognitive coefficient
        c2 = 1.5 # Social coefficient

        # Initialize swarm
        X = lo + (hi - lo) * np.random.random((swarm_size, dim))
        V = np.zeros((swarm_size, dim))
        
        # Evaluate initial swarm
        F = np.array([problem(x) for x in X])
        used = swarm_size

        # Initialize archive and pbest
        # Archive is a list of (x, f) pairs
        archive = []
        pbest_X = X.copy()
        pbest_F = F.copy()

        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def crowding_distance(points):
            """Calculate crowding distance for a set of points in objective space."""
            n = len(points)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            for m in range(n_obj):
                # Sort indices by m-th objective
                sorted_indices = sorted(range(n), key=lambda i: points[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = points[sorted_indices[0]][m]
                f_max = points[sorted_indices[-1]][m]
                
                if f_max - f_min == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (points[sorted_indices[i + 1]][m] - points[sorted_indices[i - 1]][m]) / (f_max - f_min)
            
            return distances

        def update_archive(new_x, new_f):
            """Update the archive with a new point."""
            nonlocal archive
            
            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                if dominates(new_f, af):
                    # New dominates archive point, discard archive point
                    continue
                elif dominates(af, new_f):
                    # Archive point dominates new, discard new
                    return
                else:
                    new_archive.append((ax, af))
            
            # Add new point
            new_archive.append((new_x, new_f))
            archive = new_archive
            
            # Trim archive if necessary
            if len(archive) > archive_size:
                # Calculate crowding distance
                f_vals = np.array([af for _, af in archive])
                cds = crowding_distance(f_vals)
                
                # Find index with minimum crowding distance
                min_idx = np.argmin(cds)
                archive.pop(min_idx)

        # Initialize archive with initial swarm
        for i in range(swarm_size):
            # Check if it's non-dominated by current archive
            is_dominated = False
            for ax, af in archive:
                if dominates(af, F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                # Remove points dominated by F[i]
                new_archive = []
                for ax, af in archive:
                    if dominates(F[i], af):
                        continue
                    new_archive.append((ax, af))
                new_archive.append((X[i], F[i]))
                archive = new_archive
                if len(archive) > archive_size:
                    f_vals = np.array([af for _, af in archive])
                    cds = crowding_distance(f_vals)
                    min_idx = np.argmin(cds)
                    archive.pop(min_idx)

        while used < budget:
            # Determine number of particles to update in this step
            # We can update all particles if budget allows
            particles_to_update = min(swarm_size, (budget - used) // 1) # 1 eval per particle
            
            if particles_to_update == 0:
                break

            for i in range(particles_to_update):
                if used >= budget:
                    break
                    
                # Select gbest from archive
                if not archive:
                    # Fallback to random
                    gbest_x = X[np.random.randint(swarm_size)]
                    gbest_f = F[np.random.randint(swarm_size)]
                else:
                    # Select based on crowding distance (sparsest region)
                    f_vals = np.array([af for _, af in archive])
                    cds = crowding_distance(f_vals)
                    # Choose the one with max crowding distance
                    gbest_idx = np.argmax(cds)
                    gbest_x = archive[gbest_idx][0]
                    gbest_f = archive[gbest_idx][1]
                
                # Update velocity
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_x - X[i])
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                used += 1
                
                # Update pbest
                if dominates(F[i], pbest_F[i]):
                    pbest_X[i] = X[i].copy()
                    pbest_F[i] = F[i].copy()
                else:
                    # If non-dominated, we can update pbest to the new point 
                    # to allow exploration, as the old one is in the archive.
                    # However, if we do this, pbest is always the current point.
                    # This turns MOPSO into a random walk with attraction to gbest.
                    # This is often effective for small budgets.
                    pbest_X[i] = X[i].copy()
                    pbest_F[i] = F[i].copy()
                
                # Update Archive
                update_archive(X[i], F[i])

        # Return the archive points as the final result
        if archive:
            result_X = np.array([ax for ax, _ in archive])
        else:
            result_X = X

        return result_X
# EVOLVE-BLOCK-END
