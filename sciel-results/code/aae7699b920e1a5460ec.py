# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using adaptive Tchebycheff decomposition with
    multi-start L-BFGS-B, dynamically targeting hypervolume gaps.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        eval_count = [0]

        def dominates(f_new, f_old):
            """Check if f_new dominates f_old."""
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def is_dominated(f_new):
            """Check if f_new is dominated by any point in archive_F."""
            for f in archive_F:
                if dominates(f, f_new):
                    return True
            return False

        def update_archive(x, f):
            """Update the non-dominated archive."""
            nonlocal archive_X, archive_F
            if is_dominated(f):
                return
            new_X = []
            new_F = []
            for x_old, f_old in zip(archive_X, archive_F):
                if not dominates(f, f_old):
                    new_X.append(x_old)
                    new_F.append(f_old)
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        def safe_eval(x):
            """Evaluate problem with bounds clipping and evaluation counting."""
            if eval_count[0] >= budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            eval_count[0] += 1
            return x_clipped, f

        def compute_hv_contribution(f):
            """Compute hypervolume contribution of point f relative to ref_point."""
            if np.any(f >= ref_point):
                return 0.0
            return np.prod(ref_point - f)

        def get_uniform_weights(n_obj, k):
            """Generate k uniform weight vectors in the simplex."""
            weights = []
            for i in range(k):
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)
            return weights

        def tchebycheff_obj(x, w, z):
            """
            Tchebycheff scalarization with augmented epsilon constraint.
            min max_i w_i * (f_i(x) - z_i) + rho * sum_i max(0, f_i(x) - epsilon_i)
            """
            f = problem(x)
            eval_count[0] += 1
            # Tchebycheff part
            tcheb_vals = w * (f - z)
            tcheb_max = np.max(tcheb_vals)
            # Augmented penalty for constraints (keep all objectives below epsilon)
            # Use ref_point as a soft upper bound
            penalty = 0.0
            for i in range(n_obj):
                if f[i] > ref_point[i] * 0.9:
                    penalty += (f[i] - ref_point[i] * 0.9)
            return tcheb_max + 1e-3 * penalty

        # Phase 1: Initial random sampling
        n_init = min(25, budget // 6)
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            if eval_count[0] >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            result = safe_eval(x)
            if result is None:
                break
            x_clipped, f = result
            if np.all(f < ref_point):
                update_archive(x_clipped, f)

        # If archive is empty, do more random sampling
        if len(archive_F) == 0:
            for _ in range(min(20, budget - eval_count[0])):
                if eval_count[0] >= budget:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                result = safe_eval(x)
                if result is None:
                    break
                x_clipped, f = result
                if np.all(f < ref_point):
                    update_archive(x_clipped, f)

        # Phase 2: Adaptive Tchebycheff decomposition with multi-start L-BFGS-B
        # Determine number of weight vectors based on remaining budget
        remaining = budget - eval_count[0]
        if remaining > 20:
            n_weights = min(12, max(4, remaining // 15))
            
            # Generate weight vectors - use a mix of uniform and targeted weights
            weights = []
            # Uniform weights for coverage
            for i in range(n_weights // 2):
                w = rng.dirichlet(np.ones(n_obj) * 0.5)  # More spread out
                weights.append(w)
            # Targeted weights toward extremes
            for i in range(n_weights - n_weights // 2):
                if i % n_obj == 0:
                    w = np.zeros(n_obj)
                    w[i % n_obj] = 0.8
                    w[(i + 1) % n_obj] = 0.1
                    w[(i + 2) % n_obj] = 0.1
                else:
                    w = rng.dirichlet(np.ones(n_obj) * 2.0)  # More peaked
                weights.append(w)
            
            # Normalize weights
            weights = [w / np.sum(w) for w in weights]
            
            # Get ideal point from archive
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                z_ideal = np.min(F_arr, axis=0)
            else:
                z_ideal = np.array([1.0, 1.0, 0.01])
            
            # For each weight vector, run multi-start L-BFGS-B
            for w in weights:
                if eval_count[0] >= budget:
                    break
                
                # Determine number of starts for this weight
                starts_remaining = budget - eval_count[0]
                n_starts = min(3, max(1, starts_remaining // 20))
                
                # Generate starting points: mix of archive points and random
                starts = []
                if archive_X:
                    # Pick diverse archive points
                    n_archive_starts = min(2, len(archive_X))
                    idx = rng.choice(len(archive_X), size=n_archive_starts, replace=False)
                    for i in idx:
                        starts.append(archive_X[i].copy())
                
                # Add random starts
                for _ in range(n_starts - len(starts)):
                    starts.append(lo + (hi - lo) * rng.random(dim))
                
                for x0 in starts:
                    if eval_count[0] >= budget:
                        break
                    
                    # Set max evaluations for this optimization run
                    max_evals = min(15, budget - eval_count[0])
                    if max_evals < 5:
                        continue
                    
                    # Create objective function wrapper
                    def obj_func(x, w=w, z_ideal=z_ideal):
                        f = problem(x)
                        eval_count[0] += 1
                        tcheb_vals = w * (f - z_ideal)
                        return np.max(tcheb_vals)
                    
                    try:
                        result = minimize(
                            obj_func,
                            x0,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 8, 'maxfun': max_evals}
                        )
                        
                        # Evaluate the result
                        if eval_count[0] < budget:
                            x_opt = np.clip(result.x, lo, hi)
                            f_opt = problem(x_opt)
                            eval_count[0] += 1
                            if np.all(f_opt < ref_point):
                                update_archive(x_opt, f_opt)
                    except Exception:
                        continue

        # Phase 3: Local refinement around best points
        if len(archive_F) > 0 and eval_count[0] < budget:
            # Sort archive by hypervolume contribution
            contributions = [compute_hv_contribution(f) for f in archive_F]
            sorted_idx = np.argsort(contributions)[::-1]
            
            # Refine top points
            n_refine = min(3, len(sorted_idx))
            for idx in sorted_idx[:n_refine]:
                if eval_count[0] >= budget:
                    break
                
                x_center = archive_X[idx]
                # Small perturbation search
                n_perturb = min(5, budget - eval_count[0])
                for _ in range(n_perturb):
                    if eval_count[0] >= budget:
                        break
                    # Gaussian perturbation
                    sigma = 0.1 * (hi - lo)
                    x_perturb = x_center + rng.normal(0, sigma, dim)
                    result = safe_eval(x_perturb)
                    if result is None:
                        break
                    x_clipped, f = result
                    if np.all(f < ref_point):
                        update_archive(x_clipped, f)

        # Phase 4: Final random sampling to fill any gaps
        while eval_count[0] < budget:
            x = lo + (hi - lo) * rng.random(dim)
            result = safe_eval(x)
            if result is None:
                break
            x_clipped, f = result
            if np.all(f < ref_point):
                update_archive(x_clipped, f)

        return archive_X
# EVOLVE-BLOCK-END
