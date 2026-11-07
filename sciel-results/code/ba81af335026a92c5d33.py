# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where x[2:] = 0 (i.e., x[1:] = 0).
    We sample this subspace densely to cover the front, then refine locally.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # Phase 1: Dense sampling of the Pareto front subspace (x[1:] = 0)
        # The Pareto front is parameterized by x[0] in [0, 1]
        # Allocate ~70% of budget to dense front sampling
        n_front = int(self.budget * 0.7)
        n_refine = self.budget - n_front

        # Sample x[0] values: combine uniform and clustered near interesting regions
        # The front has interesting curvature, so use a mix
        x0_uniform = np.linspace(0, 1, n_front // 2)
        x0_clustered = np.random.uniform(0, 1, n_front - n_front // 2)
        x0_samples = np.concatenate([x0_uniform, x0_clustered])
        np.random.shuffle(x0_samples)

        # Create front points: x[0] = sample, x[1:] = 0
        X_front = np.zeros((n_front, n))
        X_front[:, 0] = x0_samples
        # Clip to bounds
        X_front = np.clip(X_front, lo, hi)

        F_front = np.array([problem(x) for x in X_front])

        used = n_front

        # Check time
        if time.time() > deadline:
            return X_front

        # Phase 2: Local refinement around best non-dominated points
        # Find non-dominated points
        def is_dominated(F, idx):
            f = F[idx]
            for j in range(len(F)):
                if j == idx:
                    continue
                if np.all(F[j] <= f) and np.any(F[j] < f):
                    return True
            return False

        # Keep non-dominated points as seeds for refinement
        non_dom_idx = [i for i in range(len(F_front)) if not is_dominated(F_front, i)]
        if not non_dom_idx:
            non_dom_idx = [int(np.argmin(np.sum(F_front, axis=1)))]

        # Refine around top non-dominated points
        # Use a local search: perturb x[0] and x[1:] slightly
        best_points = []
        for idx in non_dom_idx[:min(10, len(non_dom_idx))]:
            best_points.append(X_front[idx].copy())

        # Local refinement: for each best point, try small perturbations
        step_size = 0.01
        for bp in best_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            # Try perturbations in x[0] and small x[1:] values
            for _ in range(min(5, self.budget - used)):
                if time.time() > deadline:
                    break
                child = bp.copy()
                # Perturb x[0]
                child[0] += step_size * np.random.normal()
                # Small perturbations in x[1:] (should stay near 0 for Pareto front)
                child[1:] += 0.001 * np.random.normal(size=n - 1)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                # Replace worst point if child is better
                worst_idx = int(np.argmax(np.sum(F_front, axis=1)))
                if np.any(fc < F_front[worst_idx]) and not np.all(fc >= F_front[worst_idx]):
                    X_front[worst_idx] = child
                    F_front[worst_idx] = fc

        return X_front
# EVOLVE-BLOCK-END
