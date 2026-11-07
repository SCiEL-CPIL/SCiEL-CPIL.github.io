"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Multi-Objective Particle Swarm Optimization (MOPSO).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Swarm size
        N = 20
        if self.budget < N:
            N = max(2, self.budget)
            
        # Initialize Particles
        X = lo + (hi - lo) * rng.random((N, self.dim))
        V = np.zeros((N, self.dim))
        
        # Evaluate Initial Population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Personal Bests
        pbest_X = X.copy()
        pbest_F = F.copy()
        
        # Archive (list of tuples (X, F))
        archive_X = []
        archive_F = []
        
        # Helper: Check if a point is dominated by any in archive
        def is_dominated_by_archive(f_new, archive_F_list):
            for f_arch in archive_F_list:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False
            
        # Helper: Add to archive
        def add_to_archive(x_new, f_new, archive_X_list, archive_F_list):
            # Check if new point is dominated by archive
            if is_dominated_by_archive(f_new, archive_F_list):
                return False
            
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F_list)):
                f_arch = archive_F_list[i]
                x_arch = archive_X_list[i]
                # If archive point dominates new point, new point is already handled (return False above)
                # If new point dominates archive point, remove archive point
                if np.all(f_new <= f_arch) and np.any(f_new < f_arch):
                    continue
                # If non-dominated, keep
                new_archive_X.append(x_arch)
                new_archive_F.append(f_arch)
            
            # Add new point
            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            
            # Update lists
            archive_X_list.clear()
            archive_X_list.extend(new_archive_X)
            archive_F_list.clear()
            archive_F_list.extend(new_archive_F)
            
            return True

        # Initialize Archive
        for i in range(N):
            add_to_archive(X[i], F[i], archive_X, archive_F)
            
        # Ideal Point (for leader selection)
        ideal_point = np.min(F, axis=0)
        range_point = np.max(F, axis=0) - ideal_point
        range_point[range_point == 0] = 1.0 # Avoid division by zero

        # MOPSO Loop
        # Total generations approx (Budget - N) / N
        # We will do one particle update per loop iteration? 
        # Standard MOPSO updates all particles in a generation.
        # Let's do full generations.
        gen = 0
        while used + N <= self.budget:
            # Update Ideal Point
            ideal_point = np.min(archive_F, axis=0) if len(archive_F) > 0 else np.min(F, axis=0)
            range_point = np.max(archive_F, axis=0) - ideal_point if len(archive_F) > 0 else np.max(F, axis=0) - ideal_point
            range_point[range_point == 0] = 1.0

            # Select Global Leader
            # Choose the point in archive closest to ideal point (normalized distance)
            if len(archive_F) > 0:
                dists = np.sum(((np.array(archive_F) - ideal_point) / range_point) ** 2, axis=1)
                leader_idx = np.argmin(dists)
                global_best_X = np.array(archive_X[leader_idx])
                global_best_F = np.array(archive_F[leader_idx])
            else:
                global_best_X = pbest_X[0]
                global_best_F = pbest_F[0]

            # Decay Inertia Weight
            # Total max gens = (200-20)/20 = 9
            # omega from 0.9 to 0.4
            omega = 0.9 - (0.5 * gen / 9.0)
            c1 = 1.5
            c2 = 1.5
            
            r1 = rng.random((N, 1))
            r2 = rng.random((N, 1))
            
            # Update Velocity
            V = omega * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (global_best_X - X)
            
            # Clamp Velocity
            # Max velocity component usually set to 0.2 * (hi - lo)
            max_v = 0.2 * (hi - lo)
            V = np.clip(V, -max_v, max_v)
            
            # Update Position
            X_new = X + V
            X_new = np.clip(X_new, lo, hi)
            
            # Evaluate
            F_new = np.array([problem(x) for x in X_new])
            used += N
            
            # Update Personal Bests
            for i in range(N):
                if np.all(F_new[i] <= pbest_F[i]) and np.any(F_new[i] < pbest_F[i]):
                    pbest_F[i] = F_new[i]
                    pbest_X[i] = X_new[i]
                elif np.all(pbest_F[i] <= F_new[i]) and np.any(pbest_F[i] < F_new[i]):
                    # Keep old pbest
                    pass
                else:
                    # Non-dominated, replace with new one (random choice or keep new)
                    # To encourage exploration, we can keep the new one if it's non-dominated
                    pbest_F[i] = F_new[i]
                    pbest_X[i] = X_new[i]
                
                # Add to Archive
                add_to_archive(X_new[i], F_new[i], archive_X, archive_F)
                
            # Update Current Population
            X = X_new
            F = F_new
            gen += 1
            
        # If budget remaining, evaluate single particles?
        while used < self.budget:
            # Pick a random particle, perturb slightly
            i = rng.integers(0, N)
            x_trial = X[i] + (hi - lo) * 0.05 * rng.random(self.dim)
            x_trial = np.clip(x_trial, lo, hi)
            f_trial = problem(x_trial)
            used += 1
            
            # Update pbest
            if np.all(f_trial <= pbest_F[i]) and np.any(f_trial < pbest_F[i]):
                pbest_F[i] = f_trial
                pbest_X[i] = x_trial
            else:
                # Non-dominated
                pbest_F[i] = f_trial
                pbest_X[i] = x_trial
                
            add_to_archive(x_trial, f_trial, archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END
