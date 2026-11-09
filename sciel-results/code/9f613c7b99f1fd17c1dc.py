import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # MOSA: Multi-Objective Simulated Annealing
        # Maintain an archive of non-dominated solutions
        # Use temperature to accept worse solutions to escape local optima

        # Reference point for hypervolume computation (for acceptance criterion)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        evals = 0

        # Helper: Check if f is dominated by any in archive_F
        def is_dominated(f_new, archive_F_arr):
            if len(archive_F_arr) == 0:
                return False
            # f_new is dominated if there exists f in archive such that f <= f_new and f != f_new
            return np.any(np.all(archive_F_arr <= f_new, axis=1) & np.any(archive_F_arr < f_new, axis=1))

        # Helper: Remove dominated solutions from archive
        def update_archive(X_list, F_list, x_new, f_new):
            # Check if new solution dominates any in archive
            new_idx = -1
            # First check if new solution is dominated by existing
            if len(F_list) > 0:
                F_arr = np.array(F_list)
                if np.any(np.all(F_arr <= f_new, axis=1) & np.any(F_arr < f_new, axis=1)):
                    return X_list, F_list  # New solution is dominated, don't add
            # Check if new solution dominates any existing
            to_remove = []
            for i, f_old in enumerate(F_list):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    to_remove.append(i)
            # Remove dominated solutions
            for i in reversed(to_remove):
                X_list.pop(i)
                F_list.pop(i)
            # Add new solution
            X_list.append(x_new)
            F_list.append(f_new)
            return X_list, F_list

        # Initial random solutions
        n_init = 20
        for _ in range(n_init):
            x = lo + np.random.rand(self.dim) * range_
            f = problem(x)
            evals += 1
            if evals >= self.budget:
                break
            archive_X, archive_F = update_archive(archive_X, archive_F, x.copy(), f.copy())

        # Temperature schedule
        T_max = 1.0
        T_min = 0.01
        total_steps = self.budget - evals

        # Current solution
        if len(archive_X) > 0:
            # Pick a random archive solution as starting point
            idx = np.random.randint(len(archive_X))
            x_current = archive_X[idx].copy()
            f_current = archive_F[idx].copy()
        else:
            x_current = lo + np.random.rand(self.dim) * range_
            f_current = problem(x_current)
            evals += 1
            archive_X.append(x_current.copy())
            archive_F.append(f_current.copy())

        # Simulated annealing loop
        while evals < self.budget:
            if time.time() > deadline:
                break

            # Temperature
            progress = (self.budget - evals) / max(1, total_steps)
            T = T_min + (T_max - T_min) * progress

            # Generate neighbor using Gaussian perturbation
            # Scale perturbation by remaining range
            sigma = 0.1 * range_
            x_neighbor = x_current + np.random.randn(self.dim) * sigma
            x_neighbor = np.clip(x_neighbor, lo, hi)

            f_neighbor = problem(x_neighbor)
            evals += 1

            # Check if neighbor is non-dominated
            neighbor_dominated = False
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                if np.any(np.all(F_arr <= f_neighbor, axis=1) & np.any(F_arr < f_neighbor, axis=1)):
                    neighbor_dominated = True

            if not neighbor_dominated:
                # Accept if non-dominated
                x_current = x_neighbor.copy()
                f_current = f_neighbor.copy()
                archive_X, archive_F = update_archive(archive_X, archive_F, x_current.copy(), f_current.copy())
            else:
                # Check if neighbor dominates current
                if np.all(f_neighbor <= f_current) and np.any(f_neighbor < f_current):
                    # Neighbor is better, accept
                    x_current = x_neighbor.copy()
                    f_current = f_neighbor.copy()
                    archive_X, archive_F = update_archive(archive_X, archive_F, x_current.copy(), f_current.copy())
                else:
                    # Use acceptance probability based on hypervolume contribution
                    # Simple criterion: accept with probability exp(-delta/T)
                    # where delta is a measure of how much worse the solution is
                    # Use a simple scalarization
                    weight = np.array([1.0, 1.0, 1.0])
                    scalar_current = np.dot(f_current, weight)
                    scalar_neighbor = np.dot(f_neighbor, weight)
                    delta = scalar_neighbor - scalar_current
                    if delta < 0:
                        # Neighbor is better in scalarized sense
                        x_current = x_neighbor.copy()
                        f_current = f_neighbor.copy()
                        archive_X, archive_F = update_archive(archive_X, archive_F, x_current.copy(), f_current.copy())
                    else:
                        # Accept with probability exp(-delta/T)
                        if T > 0:
                            prob = np.exp(-delta / T)
                        else:
                            prob = 0.0
                        if np.random.random() < prob:
                            x_current = x_neighbor.copy()
                            f_current = f_neighbor.copy()

            # Occasionally jump to a random archive solution to maintain diversity
            if len(archive_X) > 1 and np.random.random() < 0.05:
                idx = np.random.randint(len(archive_X))
                x_current = archive_X[idx].copy()
                f_current = archive_F[idx].copy()

            # Occasionally do a larger perturbation
            if np.random.random() < 0.1:
                x_neighbor = x_current + np.random.randn(self.dim) * 0.5 * range_
                x_neighbor = np.clip(x_neighbor, lo, hi)
                f_neighbor = problem(x_neighbor)
                evals += 1
                if evals >= self.budget:
                    break

                neighbor_dominated = False
                if len(archive_F) > 0:
                    F_arr = np.array(archive_F)
                    if np.any(np.all(F_arr <= f_neighbor, axis=1) & np.any(F_arr < f_neighbor, axis=1)):
                        neighbor_dominated = True

                if not neighbor_dominated:
                    x_current = x_neighbor.copy()
                    f_current = f_neighbor.copy()
                    archive_X, archive_F = update_archive(archive_X, archive_F, x_current.copy(), f_current.copy())
                elif np.all(f_neighbor <= f_current) and np.any(f_neighbor < f_current):
                    x_current = x_neighbor.copy()
                    f_current = f_neighbor.copy()
                    archive_X, archive_F = update_archive(archive_X, archive_F, x_current.copy(), f_current.copy())

        return archive_X
