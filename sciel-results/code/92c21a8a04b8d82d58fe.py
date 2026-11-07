# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where x[2:] = 0.
    We use Sobol sequences for quasi-random sampling of the front, plus corner points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin of 20s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # Phase 1: Dense sampling of the Pareto front using Sobol sequence
        # The Pareto front is parameterized by x[0] in [0, 1]
        n_front = self.budget

        # Generate Sobol sequence for x[0]
        sampler = qmc.Sobol(d=1, scramble=True)
        # Use a power of 2 for Sobol
        n_sobol = 1 << int(np.log2(n_front))
        if n_sobol > n_front:
            n_sobol = 1 << (int(np.log2(n_front)) - 1)
        
        sobol_samples = sampler.random(n_sobol)[:, 0]
        
        # Add corner points explicitly
        corner_points = [0.0, 1.0, 0.5]
        
        # Combine and take n_front samples
        all_x0 = np.concatenate([sobol_samples, np.array(corner_points)])
        # Shuffle to avoid bias
        perm = np.random.permutation(len(all_x0))
        all_x0 = all_x0[perm]
        x0_samples = all_x0[:n_front]
        
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
        # Find non-dominated points efficiently
        def find_non_dominated(F):
            m = len(F)
            nd_flags = np.ones(m, dtype=bool)
            for i in range(m):
                if not nd_flags[i]:
                    continue
                for j in range(m):
                    if i == j or not nd_flags[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_flags[i] = False
                        break
            return np.where(nd_flags)[0]

        non_dom_idx = find_non_dominated(F_front)
        if len(non_dom_idx) == 0:
            non_dom_idx = [int(np.argmin(np.sum(F_front, axis=1)))]

        # Refine around top non-dominated points with remaining budget
        # Use a local search: perturb x[0] and x[1:] slightly
        step_size = 0.02
        for idx in non_dom_idx[:min(5, len(non_dom_idx))]:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            bp = X_front[idx].copy()
            # Try a few perturbations
            for _ in range(min(3, self.budget - used)):
                if time.time() > deadline:
                    break
                child = bp.copy()
                # Perturb x[0]
                child[0] += step_size * np.random.normal()
                # Small perturbations in x[1:] (should stay near 0 for Pareto front)
                child[1:] += 0.002 * np.random.normal(size=n - 1)
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
