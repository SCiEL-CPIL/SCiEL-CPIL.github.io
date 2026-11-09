import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Multi-objective optimizer for RE34: Sobol init + HV-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Store ALL evaluated points - every evaluation counts
        X_all = []
        F_all = []

        def add_point(x, f):
            X_all.append(np.asarray(x, dtype=float))
            F_all.append(np.asarray(f, dtype=float))

        # ---------- Hypervolume helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            if len(sub) == 0:
                return 0.0
            sub = sub[np.all(sub < ref2, axis=1)]
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                w0 = min(x, ref2[0]) - prev_x
                if w0 <= 0:
                    continue
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                if ybest >= ref2[1]:
                    prev_x = x
                    continue
                total += w0 * (ref2[1] - ybest)
                prev_x = x
            return total

        def hv3(P, ref3):
            """3D hypervolume (minimization)."""
            if len(P) == 0:
                return 0.0
            mask = np.all(P < ref3, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w0 = min(x, ref3[0]) - prev_x
                if w0 <= 0:
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev_x = x
            return total

        def pareto_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            keep = []
            for i in range(n):
                fi = F[i]
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        # ---------- Sobol initialization for first 128 evaluations ----------
        n_sobol = min(128, budget)
        # Use scipy's qmc.Sobol for deterministic space-filling sequence
        from scipy.integrate import quad
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=dim, scramble=False)
            sobol_samples = sobol.random(n_sobol)
        except ImportError:
            # Fallback to random if Sobol not available
            sobol_samples = rng.random((n_sobol, dim))
        
        for i in range(n_sobol):
            if len(X_all) >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            x = lo + (hi - lo) * sobol_samples[i]
            x = np.clip(x, lo, hi)
            add_point(x, problem(x))

        used = len(X_all)

        # ---------- Main search loop: HV-guided local search on top points ----------
        # For remaining evaluations, do local refinement of top-10 HV contributors
        while used < budget:
            if time.time() > deadline - 10.0:
                break
            
            F_arr = np.array(F_all)
            current_hv = hv3(F_arr, ref)
            
            # Compute marginal HV contribution for each point
            n_points = len(F_all)
            if n_points < 2:
                x = lo + (hi - lo) * rng.random(dim)
                fc = problem(x)
                add_point(x, fc)
                used += 1
                continue
            
            # Approximate marginal contribution: HV with point vs without
            # Use a fast approximation: rank by distance to reference in objective space
            # and select top candidates for local search
            
            # Select top 10 non-dominated points by proximity to ideal point
            keep = pareto_indices(F_arr)
            if len(keep) == 0:
                keep = list(range(n_points))
            
            # Weight by how much each point contributes to HV (approximate by objective sum)
            cand_F = F_arr[keep]
            # Normalize objectives to [0,1] range based on current bounds
            f_min = cand_F.min(axis=0)
            f_max = cand_F.max(axis=0)
            f_range = f_max - f_min + 1e-9
            normalized = (cand_F - f_min) / f_range
            # Lower normalized sum = better
            sums = np.sum(normalized, axis=1)
            
            # Take top 10 (lowest sums)
            top_k = min(10, len(keep))
            top_indices = np.argsort(sums)[:top_k]
            top_parent_indices = [keep[i] for i in top_indices]
            
            # For each top point, try small axis-aligned perturbations
            for pidx in top_parent_indices:
                if used >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break
                
                # Try perturbing each dimension
                for d in range(dim):
                    if used >= budget:
                        break
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Small step size
                    step_size = 0.05 * (hi[d] - lo[d])
                    
                    # Try + and - perturbations
                    for sign in [1, -1]:
                        if used >= budget:
                            break
                        if time.time() > deadline - 10.0:
                            break
                        
                        child = X_all[pidx].copy()
                        child[d] += sign * step_size
                        child = np.clip(child, lo, hi)
                        
                        fc = problem(child)
                        add_point(child, fc)
                        used += 1
                        
                        # If this improved things, stop trying other directions for this dim
                        # (simple heuristic: check if new point is non-dominated)
                        if len(F_all) > 0:
                            F_new = np.array(F_all[-1:])
                            F_existing = np.array(F_all[:-1])
                            if len(F_existing) > 0:
                                is_nondominated = True
                                for j in range(len(F_existing)):
                                    if np.all(F_existing[j] <= F_new[0]) and np.any(F_existing[j] < F_new[0]):
                                        is_nondominated = False
                                        break
                                if is_nondominated:
                                    break  # Good improvement, move to next dimension

        return X_all
