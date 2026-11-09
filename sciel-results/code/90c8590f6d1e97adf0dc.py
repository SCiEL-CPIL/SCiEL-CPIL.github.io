"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Multi-Objective Particle Swarm Optimization (MOPSO)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use the global numpy random state as seeded by the evaluator
        rng = np.random

        # MOPSO Parameters
        # Population size needs to be small to fit within budget
        # Budget 200. Let's use pop_size=10, giving ~20 iterations.
        pop_size = 10
        
        # Inertia weight
        w = 0.7
        # Cognitive coefficient
        c1 = 1.5
        # Social coefficient
        c2 = 1.5
        
        # Velocity limit (usually a fraction of the search space span)
        vmax = 0.2 * span
        
        # Initialize positions uniformly
        X = lo + span * rng.random((pop_size, dim))
        
        # Initialize velocities to zero
        V = np.zeros((pop_size, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Initialize personal bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # External Archive
        # We will store indices or copies of non-dominated solutions
        # For simplicity, we store the objective vectors and decision vectors
        archive_X = []
        archive_F = []
        
        def is_dominated(a, b):
            """Return True if a dominates b (a is better than b in all objectives, strict in at least one)."""
            return np.all(a <= b) and np.any(a < b)
        
        def update_archive(x, f):
            """Add x, f to archive if it is non-dominated, removing any dominated points."""
            non_dominated = True
            # Check if new point is dominated by any in archive
            for af in archive_F:
                if is_dominated(af, f):
                    non_dominated = False
                    break
            
            if non_dominated:
                # Remove any points in archive dominated by the new point
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not is_dominated(f, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                
                new_archive_X.append(x)
                new_archive_F.append(f)
                return new_archive_X, new_archive_F
            
            return archive_X, archive_F

        # Initialize archive with initial population
        for i in range(pop_size):
            archive_X, archive_F = update_archive(X[i], F[i])
        
        # Main Loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate crowding distance for archive to select gbest
            # Simple approximation: normalize objectives and sum distances
            if len(archive_F) > 1:
                # Normalize archive objectives
                af_arr = np.array(archive_F)
                min_f = af_arr.min(axis=0)
                max_f = af_arr.max(axis=0)
                range_f = max_f - min_f
                range_f[range_f == 0] = 1.0
                norm_af = (af_arr - min_f) / range_f
                
                # Calculate crowding distance
                n_arch = len(archive_F)
                crowding = np.zeros(n_arch)
                for k in range(n_obj):
                    # Sort indices by k-th objective
                    idx_sorted = np.argsort(norm_af[:, k])
                    crowding[idx_sorted[0]] = np.inf
                    crowding[idx_sorted[-1]] = np.inf
                    for j in range(1, n_arch - 1):
                        prev_idx = idx_sorted[j-1]
                        next_idx = idx_sorted[j+1]
                        dist = norm_af[next_idx, k] - norm_af[prev_idx, k]
                        crowding[idx_sorted[j]] += dist
            else:
                crowding = np.ones(len(archive_F))
            
            # Select gbest for each particle
            # Use tournament selection or random selection based on crowding
            # To ensure diversity, we can pick randomly from the archive, 
            # or use a probability proportional to crowding distance.
            # Let's use a simple random selection from the archive to avoid bias,
            # or better: select the one with the highest crowding distance (most isolated).
            # Actually, standard MOPSO often uses a random selection from the archive 
            # to prevent premature convergence to a single point.
            
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                # If time is running out, break
                if time.time() > deadline - 5.0:
                    break
                
                # Select gbest from archive
                # Random selection from archive
                if len(archive_F) > 0:
                    gbest_idx = rng.integers(len(archive_F))
                    gbest_X = archive_X[gbest_idx]
                else:
                    gbest_X = Pbest_X[i]
                
                # Update velocity
                r1 = rng.random()
                r2 = rng.random()
                
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -vmax, vmax)
                
                # Update position
                new_X[i] = X[i] + V[i]
                
                # Clip position to bounds
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Evaluate
                new_F[i] = problem(new_X[i])
                used += 1
                
                # Update personal best
                # If new_F[i] dominates Pbest_F[i], update
                if is_dominated(new_F[i], Pbest_F[i]):
                    Pbest_X[i] = new_X[i]
                    Pbest_F[i] = new_F[i]
                
                # Update archive
                archive_X, archive_F = update_archive(new_X[i], new_F[i])
                
                # Update current position and fitness for next iteration
                X[i] = new_X[i]
                F[i] = new_F[i]
            
            # Optional: If stagnation is detected (no archive update in last few iterations),
            # we could reset some velocities or positions. 
            # For now, rely on the randomness in gbest selection.
            
        return X
# EVOLVE-BLOCK-END
