import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive refinement for f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, f0, max_iter=30, initial_step=0.3):
            nonlocal calls_made
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            for i in range(1, dim + 1):
                if not time_ok():
                    return x0, f0
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step
                fvals[i] = safe_call(simplex[i])

            for _ in range(max_iter):
                if not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.std(fvals) < 1e-8 * (1.0 + abs(fvals[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def coordinate_descent(x, f, max_pass=6, step=0.5):
            nonlocal calls_made
            for _ in range(max_pass):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sign in (1.0, -1.0):
                        if not time_ok():
                            break
                        cand = x.copy()
                        cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                        fc = safe_call(cand)
                        if fc < f - 1e-12:
                            x = cand
                            f = fc
                            improved = True
                            break
                if not improved:
                    break
            return x, f

        # Phase 1: Latin Hypercube Sampling for global coverage
        n_samples = 50
        rng = np.random.RandomState(42)
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = -5.0 + (perm + 0.5) * (10.0 / n_samples)
        
        # Evaluate LHS points and store results
        lhs_results = []
        for p in points:
            if not time_ok():
                break
            # We need the value to sort. safe_call returns it.
            # But safe_call also updates global best. That's fine.
            f_val = safe_call(p)
            if np.isfinite(f_val):
                lhs_results.append((f_val, p.copy()))

        # Sort by function value
        lhs_results.sort(key=lambda item: item[0])
        
        # Phase 2: Local refinement from top K points
        # Select top 3 unique points (by value, though duplicates are unlikely in LHS)
        top_k = 3
        candidates = []
        seen = set()
        for f_val, x in lhs_results:
            # Simple uniqueness check by rounding to 2 decimals to avoid near-duplicates if any
            key = tuple(np.round(x, 2))
            if key not in seen:
                seen.add(key)
                candidates.append((f_val, x))
            if len(candidates) >= top_k:
                break
        
        for f_val, x in candidates:
            if not time_ok():
                break
            # Run Nelder-Mead from this candidate
            x_opt, f_opt = nelder_mead(x, f_val, max_iter=40, initial_step=0.5)
            # Update global best if NM found something better
            # safe_call inside NM already updated best_x/best_f if it found better points
            # But we should ensure we check the final result of NM
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: Final coordinate descent polish
        if time_ok() and best_x is not None:
            best_x, best_f = coordinate_descent(best_x, best_f, max_pass=10, step=0.5)

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
