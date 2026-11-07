"""Multi-objective optimizer for RE37: MOEA/D with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D multi-objective optimizer with Tchebycheff decomposition and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin well before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Generate weight vectors on 2D simplex (for 3 objectives) ---
        # Use a systematic grid to ensure good coverage
        n_per_axis = 5  # 5 points per axis on the simplex
        weights = []
        for i in range(n_per_axis + 1):
            for j in range(n_per_axis + 1 - i):
                k = n_per_axis - i - j
                w = np.array([i, j, k], dtype=float) / n_per_axis
                weights.append(w)
        weights = np.array(weights)  # shape (M, n_obj)
        M = len(weights)

        # --- Neighborhood: each subproblem connects to its closest neighbors ---
        # For a grid on the simplex, use a fixed neighborhood size
        T = 20  # neighborhood size
        neighbors = []
        for i in range(M):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            dists[i] = np.inf
            idx = np.argsort(dists)[:T]
            neighbors.append(idx)
        neighbors = np.array(neighbors)

        # --- Initialize population ---
        rng = np.random.default_rng()
        X = lo + (hi - lo) * rng.random((M, d))
        F = np.array([problem(x) for x in X])
        used = M

        # Ideal and nadir points (updated dynamically)
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)

        # Tchebycheff scalarization
        def tchebycheff(f, w, ideal_pt, nadir_pt):
            # Normalize objectives
            span = nadir_pt - ideal_pt
            span[span < 1e-12] = 1.0
            f_norm = (f - ideal_pt) / span
            # Tchebycheff: max_i w_i * |f_i - z_i|
            return np.max(w * f_norm)

        # Compute initial fitness
        def compute_fitness():
            fit = np.zeros(M)
            for i in range(M):
                fit[i] = tchebycheff(F[i], weights[i], ideal, nadir)
            return fit

        fit = compute_fitness()

        # --- Main loop ---
        while used < budget:
            # Deadline check
            if time.time() > deadline:
                break

            # Update ideal and nadir
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
            fit = compute_fitness()

            # For each subproblem, select a parent from its neighborhood
            for i in range(M):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parent from neighborhood using tournament selection
                nb = neighbors[i]
                # Tournament of 2
                p1_idx, p2_idx = rng.integers(0, len(nb), size=2)
                p1, p2 = nb[p1_idx], nb[p2_idx]
                parent = p1 if fit[p1] <= fit[p2] else p2

                # Mutation: adaptive step size
                progress = used / budget
                step = 0.3 * (1.0 - 0.5 * progress)
                child = X[parent] + step * (hi - lo) * rng.standard_normal(d)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update ideal/nadir before evaluating replacement
                # Try to replace the worst in neighborhood
                nb = neighbors[i]
                # Find the worst neighbor by fitness
                nb_fits = fit[nb]
                worst_local = nb[np.argmax(nb_fits)]

                # Also consider replacing subproblem i itself if it's worse
                # Accept if child is better than worst neighbor or subproblem i
                child_fit_val = tchebycheff(fc, weights[i], ideal, nadir)

                # Replace if better than worst in neighborhood
                if child_fit_val < fit[worst_local]:
                    X[worst_local] = child
                    F[worst_local] = fc
                elif child_fit_val < fit[i]:
                    X[i] = child
                    F[i] = fc
                else:
                    # With some probability, still replace to maintain diversity
                    if rng.random() < 0.1:
                        X[i] = child
                        F[i] = fc

                # Update ideal and nadir
                ideal = np.min(F, axis=0)
                nadir = np.max(F, axis=0)

        return X
# EVOLVE-BLOCK-END
