"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer: Sobol initialization + non-dominated mutation + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Sobol quasi-random initialization for space-filling coverage
        # Use ~50% of budget for initial exploration
        init_size = min(100, max(20, budget // 2))
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
            sobol_samples = sampler.random(n=init_size)
        except Exception:
            sobol_samples = np.random.random((init_size, dim))
        
        X = lo + sobol_samples * (hi - lo)
        F = np.array([problem(x) for x in X])
        evals = init_size

        def is_dominated(F_mat, idx):
            """Check if point idx is dominated by any other point."""
            fi = F_mat[idx]
            for j in range(len(F_mat)):
                if j == idx:
                    continue
                if np.all(F_mat[j] <= fi) and np.any(F_mat[j] < fi):
                    return True
            return False

        # Phase 2: Evolutionary search - mutate non-dominated parents
        # Use remaining budget for iterative improvement
        remaining = budget - evals
        
        # Adaptive sigma: start larger, shrink over time
        num_rounds = min(remaining, 100)
        points_per_round = max(1, remaining // max(1, num_rounds))
        
        round_idx = 0
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Find current non-dominated points
            non_dom_indices = [i for i in range(len(F)) if not is_dominated(F, i)]
            
            if not non_dom_indices:
                # Fallback: use all points
                non_dom_indices = list(range(len(F)))
            
            # Progress-based sigma
            progress = round_idx / max(1, num_rounds - 1)
            sigma = 0.15 * (1.0 - 0.7 * progress)
            
            # Select which non-dominated points to perturb
            n_perturb = min(len(non_dom_indices), points_per_round)
            if n_perturb > 0:
                selected = np.random.choice(non_dom_indices, size=n_perturb, replace=False)
                
                new_points = []
                for idx in selected:
                    base = X[idx]
                    perturbed = base + sigma * np.random.standard_normal(dim)
                    perturbed = np.clip(perturbed, lo, hi)
                    new_points.append(perturbed)
                
                if new_points:
                    new_points = np.array(new_points)
                    new_F = np.array([problem(x) for x in new_points])
                    X = np.vstack([X, new_points])
                    F = np.vstack([F, new_F])
                    evals += len(new_points)
            
            round_idx += 1

        # Phase 3: Local refinement around best non-dominated points
        remaining = budget - evals
        if remaining > 0 and len(F) > 0:
            # Find non-dominated points
            non_dom_indices = [i for i in range(len(F)) if not is_dominated(F, i)]
            
            if non_dom_indices:
                # Sort by sum of objectives (lower is better)
                sums = F[non_dom_indices].sum(axis=1)
                sorted_non_dom = [non_dom_indices[i] for i in np.argsort(sums)]
                
                # Refine the top points with smaller sigma
                n_refine = min(len(sorted_non_dom), max(1, remaining // 5))
                for idx in sorted_non_dom[:n_refine]:
                    if evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Multiple small perturbations
                    for _ in range(5):
                        if evals >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        base = X[idx]
                        sigma_local = 0.03
                        perturbed = base + sigma_local * np.random.standard_normal(dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        
                        fc = problem(perturbed)
                        X = np.vstack([X, perturbed])
                        F = np.vstack([F, fc.reshape(1, -1)])
                        evals += 1

        return X
# EVOLVE-BLOCK-END
