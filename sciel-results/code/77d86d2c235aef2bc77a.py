"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: random init, adaptive mutation, hypervolume-aware replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume (from task contract)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Population size: balance between exploration and exploitation
        pop_size = min(50, max(20, self.budget // 4))

        # Initialize population with mix of random and center-biased points
        # WFG4 Pareto front is typically near the center of the search space
        X = np.empty((pop_size, dim))
        # Half random, half center-biased
        n_rand = pop_size // 2
        n_center = pop_size - n_rand

        X[:n_rand] = lo + span * np.random.rand(n_rand, dim)
        if n_center > 0:
            center_samples = np.clip(0.5 + 0.15 * np.random.randn(n_center, dim), 0, 1)
            X[n_rand:] = lo + span * center_samples

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume calculation
        all_F = [F.copy()]

        def dominated_mask(F_arr):
            """Return boolean mask where True means the point is dominated."""
            n = len(F_arr)
            mask = np.zeros(n, dtype=bool)
            for i in range(n):
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] and F[j] != F[i]
                dominates_i = np.all(F_arr <= F_arr[i], axis=1) & np.any(F_arr < F_arr[i], axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    mask[i] = True
            return mask

        def select_replacement_idx(F_arr, fc, dominated):
            """Select which index to replace.
            Prefer replacing a point dominated by the child.
            Otherwise, replace the point with worst hypervolume contribution."""
            if np.any(dominated):
                # Replace a random dominated point
                dom_indices = np.where(dominated)[0]
                return dom_indices[np.random.randint(len(dom_indices))]
            else:
                # Replace the point with the largest sum (worst in magnitude)
                return int(np.argmax(np.sum(F_arr, axis=1)))

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated parents
            dom_mask = dominated_mask(F)
            nd_indices = np.where(~dom_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Select a parent from non-dominated set
            parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            parent = X[parent_idx]

            # Adaptive step size: larger early, smaller later
            progress = used / self.budget
            step_size = 0.5 * (1 - progress) + 0.05

            # Mutate
            noise = np.random.randn(dim)
            child = parent + step_size * span * noise
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Determine replacement
            # Check if child dominates any existing point
            child_dominates = np.all(F >= fc, axis=1) & np.any(F > fc, axis=1)
            if np.any(child_dominates):
                replace_idx = int(np.where(child_dominates)[0][np.random.randint(np.sum(child_dominates))])
            else:
                replace_idx = select_replacement_idx(F, fc, dom_mask)

            X[replace_idx] = child
            F[replace_idx] = fc

            # Record all evaluated points
            all_F[-1][replace_idx] = fc

            # Every 50 evaluations, add a new snapshot or just keep accumulating
            # For efficiency, we'll just keep the current F as part of all_F
            if used % 50 == 0 and len(all_F) < 10:
                all_F.append(F.copy())

        # Final snapshot
        all_F.append(F.copy())

        return X
# EVOLVE-BLOCK-END
