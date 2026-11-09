"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using a decomposition-guided local search strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG7-aware initialization: spread shape variables, random position variables
        rng = np.random.default_rng(0)
        N_init = 12
        X = np.empty((N_init, dim))
        for i in range(N_init):
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * (0.1 + 0.8 * rng.random())
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * (0.1 + 0.8 * rng.random())
            X[i, 2] = lo[2] + (hi[2] - lo[2]) * (0.1 + 0.8 * rng.random())
            X[i, 3:] = lo[3:] + (hi[3:] - lo[3:]) * rng.random(dim - 3)

        F = np.array([problem(x) for x in X])
        evals = N_init

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Keep track of all evaluated points for Pareto front
        all_X = X.copy()
        all_F = F.copy()

        # Weight vectors for decomposition (simplex for 3 objectives)
        n_weights = 10
        weights = []
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                w = np.array([i, j, k], dtype=float) / n_weights
                weights.append(w)
        weights = np.array(weights)

        # Local search using scipy minimize on aggregated objectives
        # We'll do a few rounds of local search from promising points
        # Use Chebyshev aggregation with different weights

        # Sort initial points by sum of objectives to find promising seeds
        sums = F.sum(axis=1)
        sorted_idx = np.argsort(sums)
        
        # Pick top few points as seeds for local search
        n_seeds = min(6, N_init)
        seeds = sorted_idx[:n_seeds]

        # For each seed, do local search with different weight vectors
        # Use Nelder-Mead or L-BFGS-B depending on dim
        for seed_idx in seeds:
            if evals >= budget or time.time() > deadline:
                break
            
            x0 = all_X[seed_idx].copy()
            f0 = all_F[seed_idx].copy()
            
            # Try a few different weight vectors for local search
            # Use the 3 corner weights and 3 edge-midpoint weights
            search_weights = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.5, 0.5, 0.0]),
                np.array([0.5, 0.0, 0.5]),
                np.array([0.0, 0.5, 0.5]),
            ]
            
            for w in search_weights:
                if evals >= budget or time.time() > deadline:
                    break
                
                # Define aggregated objective
                def agg_obj(x):
                    nonlocal evals
                    fx = problem(np.clip(x, lo, hi))
                    evals += 1
                    # Chebyshev aggregation
                    return np.max(w * fx)
                
                # Use L-BFGS-B with bounds
                try:
                    res = minimize(agg_obj, x0, method='L-BFGS-B', 
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 5, 'maxfev': 5})
                    if res.success or res.fun < np.max(w * f0):
                        x_new = np.clip(res.x, lo, hi)
                        f_new = problem(x_new)
                        evals += 1
                        all_X = np.vstack([all_X, x_new])
                        all_F = np.vstack([all_F, f_new])
                except:
                    pass

        # Add some random perturbations around best points to diversify
        # Find non-dominated points
        def is_non_dominated(F_set):
            n = len(F_set)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        nd[i] = False
                        break
            return nd

        # Get non-dominated set
        nd_mask = is_non_dominated(all_F)
        nd_X = all_X[nd_mask]
        nd_F = all_F[nd_mask]

        # If we have budget left, do a final round of random + local refinement
        if evals < budget and len(nd_X) > 0:
            # Pick a few non-dominated points and perturb them
            n_perturb = min(5, len(nd_X), (budget - evals) // 3)
            for _ in range(n_perturb):
                if evals >= budget or time.time() > deadline:
                    break
                idx = rng.integers(len(nd_X))
                x_base = nd_X[idx].copy()
                # Gaussian perturbation
                sigma = 0.05 * (hi - lo)
                x_pert = x_base + rng.normal(0, sigma)
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = problem(x_pert)
                evals += 1
                all_X = np.vstack([all_X, x_pert])
                all_F = np.vstack([all_F, f_pert])

        return all_X
# EVOLVE-BLOCK-END
