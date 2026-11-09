"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution-style multi-objective optimizer with archive-guided search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # Structured initialization: corners, midpoints, and random
        pop_size = min(35, max(15, budget // 5))
        
        # Generate corner points
        corners = []
        for i in range(2 ** dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(corner)
        
        # Midpoints and structured points
        structured = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
            lo + 0.5 * span,
        ]
        
        # Build initial population: structured first, then random to fill
        init_points = []
        # Add all corners if they fit, otherwise sample
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_indices = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_indices])
        
        # Add structured midpoints
        for sp in structured:
            if len(init_points) < pop_size:
                init_points.append(sp)
        
        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))
        
        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive management
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any archive member
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove archive members dominated by new point
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # DE parameters
        CR = 0.7
        F_mut_start = 0.5
        F_mut_end = 0.9

        while used < budget:
            if time.time() - start_time > DEADLINE:
                break

            # Adaptive mutation scale: increase over time to explore more
            progress = used / budget
            F_mut = F_mut_start + (F_mut_end - F_mut_start) * progress

            # Select parents: prefer archive if it has enough members
            if len(archive_X) >= 3:
                idxs = rng.choice(len(archive_X), size=3, replace=False)
                x1 = archive_X[idxs[0]].copy()
                x2 = archive_X[idxs[1]].copy()
                x3 = archive_X[idxs[2]].copy()
            elif len(archive_X) >= 1:
                # Use archive for one parent, population for others
                a_idx = rng.integers(len(archive_X))
                x1 = archive_X[a_idx].copy()
                p_idxs = rng.choice(pop_size, size=2, replace=False)
                x2 = X[p_idxs[0]].copy()
                x3 = X[p_idxs[1]].copy()
            else:
                p_idxs = rng.choice(pop_size, size=3, replace=False)
                x1 = X[p_idxs[0]].copy()
                x2 = X[p_idxs[1]].copy()
                x3 = X[p_idxs[2]].copy()

            # DE/rand/1 mutation
            mutant = x1 + F_mut * (x2 - x3)
            mutant = np.clip(mutant, lo, hi)

            # Select base vector from population
            base_idx = rng.integers(pop_size)

            # Binomial crossover
            j_rand = rng.integers(dim)
            trial = X[base_idx].copy()
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Update archive
            update_archive(trial, f_trial)

            # Population replacement: use dominance count
            # Count how many population members dominate each other
            dominance_count = np.zeros(pop_size)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominance_count[j] += 1

            worst_idx = np.argmax(dominance_count)
            
            # Replace if trial is not dominated by the worst, or if worst is very bad
            if dominance_count[worst_idx] > 0:
                # Trial replaces the most-dominated individual
                X[worst_idx] = trial
                F[worst_idx] = f_trial
            else:
                # All non-dominated: replace if trial has better objective sum
                if np.sum(f_trial) < np.sum(F[base_idx]):
                    X[base_idx] = trial
                    F[base_idx] = f_trial
                elif rng.random() < 0.3:
                    # Random replacement to maintain diversity
                    X[base_idx] = trial
                    F[base_idx] = f_trial

            # Periodic diversity injection
            if used % 15 == 0 and used < budget - 5:
                if time.time() - start_time > DEADLINE:
                    break
                for k in range(2):
                    if used >= budget:
                        break
                    rand_x = lo + span * rng.random(dim)
                    rand_f = np.array(problem(rand_x))
                    used += 1
                    update_archive(rand_x, rand_f)
                    # Replace a random population member
                    r_idx = rng.integers(pop_size)
                    X[r_idx] = rand_x
                    F[r_idx] = rand_f

        return X
# EVOLVE-BLOCK-END
