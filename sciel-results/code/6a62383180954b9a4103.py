"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling followed by Nelder-Mead refinement on top HV contributors."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol sequence sampling - use 160 points for better initial coverage
        n_sobol = 160
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)
        X_sobol = lo + sobol_samples * span
        X_sobol = np.clip(X_sobol, lo, hi)

        # Evaluate Sobol samples
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        # Combine into a single archive
        X_all = X_sobol.copy()
        F_all = F_sobol.copy()

        # Reference point for HV calculation
        ref_point = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points dominated by ref (minimization)."""
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_f2 = -np.inf
            for i in range(len(pts)):
                f1, f2 = pts[i]
                if f2 < max_f2:
                    hv += (ref[0] - f1) * (max_f2 - f2) if max_f2 > f2 else 0.0
                else:
                    max_f2 = f2
                    hv += (ref[0] - f1) * (ref[1] - f2)
            return hv

        def get_pareto_front(points):
            """Return indices of non-dominated points."""
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                        dominated[j] = True
                    elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def hv_contribution(idx, points, ref):
            """Compute the hypervolume contribution of point idx to the set."""
            full_hv = hypervolume_2d(points, ref)
            remaining = np.delete(points, idx, axis=0)
            if len(remaining) == 0:
                return full_hv
            reduced_hv = hypervolume_2d(remaining, ref)
            return full_hv - reduced_hv

        # Phase 2: Get Pareto front and identify top contributors
        pareto_idx = get_pareto_front(F_all)
        if len(pareto_idx) == 0:
            pareto_idx = np.array([0])
        
        # Compute contributions only for Pareto front points
        pareto_points = F_all[pareto_idx]
        n_pareto = len(pareto_idx)
        contributions = np.array([hv_contribution(i, F_all, ref_point) for i in pareto_idx])
        
        # Select top 12 contributors (we have 40 evals remaining)
        n_top = min(12, n_pareto)
        top_local_idx = np.argsort(contributions)[::-1][:n_top]
        top_indices = pareto_idx[top_local_idx]
        
        remaining_budget = budget - used
        
        if remaining_budget > 0:
            # Create diverse weightings based on Pareto front structure
            if n_pareto >= 2:
                f1_min = np.min(pareto_points[:, 0])
                f1_max = np.max(pareto_points[:, 0])
                f2_min = np.min(pareto_points[:, 1])
                f2_max = np.max(pareto_points[:, 1])
                
                # Generate weights that span the trade-off
                n_weights = n_top
                if n_weights >= 2:
                    # Include extreme weights and intermediate ones
                    w1_vals = np.linspace(0.0, 1.0, n_weights)
                    weights = []
                    for w1 in w1_vals:
                        if w1 < 1e-6:
                            weights.append(np.array([0.0, 1.0]))
                        elif 1 - w1 < 1e-6:
                            weights.append(np.array([1.0, 0.0]))
                        else:
                            # Normalize so they sum to 1
                            weights.append(np.array([w1, 1 - w1]))
                else:
                    weights = [np.array([1.0, 1.0])]
            else:
                weights = [np.array([1.0, 1.0])]
            
            # Allocate evals: give more to high-contribution points
            # Use at most 15 evals per NM run
            max_evals_per_nm = 15
            min_evals_per_nm = 5
            total_available = remaining_budget
            n_runs = n_top
            
            # Distribute budget: higher contribution gets more evals
            contrib_norm = contributions[top_local_idx] / (np.sum(contributions[top_local_idx]) + 1e-10)
            raw_alloc = contrib_norm * total_available
            evals_per_run = np.clip(raw_alloc, min_evals_per_nm, max_evals_per_nm).astype(int)
            
            # Ensure we don't exceed budget
            while np.sum(evals_per_run) > total_available:
                max_idx = np.argmax(evals_per_run)
                if evals_per_run[max_idx] > min_evals_per_nm:
                    evals_per_run[max_idx] -= 1
                else:
                    break
            
            for i, idx in enumerate(top_indices):
                if used >= budget:
                    break
                
                x0 = X_all[idx].copy()
                w = weights[i % len(weights)]
                maxfev = int(evals_per_run[i])
                
                if maxfev < 3:
                    maxfev = 3
                
                def objective(x):
                    nonlocal used
                    if used >= budget:
                        return 0.0
                    fx = problem(x)
                    used += 1
                    return np.dot(fx, w)
                
                options = {
                    'maxiter': maxfev,
                    'maxfev': maxfev,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'disp': False
                }
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                    if result.fun is not None and used < budget:
                        x_best = np.clip(result.x, lo, hi)
                        f_best = problem(x_best)
                        used += 1
                        X_all = np.vstack([X_all, x_best])
                        F_all = np.vstack([F_all, f_best])
                except Exception:
                    pass
        
        return X_all
# EVOLVE-BLOCK-END
