"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start with stratified sampling + BFGS local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        def is_time_up():
            return time.time() > deadline

        # Generate stratified sample points (Latin Hypercube)
        def generate_lhs(n_points, dim, rng):
            """Generate n_points using Latin Hypercube sampling in [-5, 5]^dim."""
            points = np.zeros((n_points, dim))
            for d in range(dim):
                perm = rng.permutation(n_points)
                # Stratified: each point gets a unique stratum
                strata = (perm + rng.uniform(0, 1, n_points)) / n_points
                points[:, d] = -5 + 10 * strata
            return points

        # BFGS local search with tight budget
        def bfgs_local(x0, f0, max_evals):
            """Run BFGS from x0, using at most max_evals function evaluations."""
            nonlocal best_x, best_f, total_calls

            if f0 is None:
                return

            # Use scipy's minimize with BFGS
            # We need to track evaluations carefully
            class EvalCounter:
                def __init__(self):
                    self.count = 0

            counter = EvalCounter()

            def wrapped_func(x):
                nonlocal best_f, best_x
                if counter.count >= max_evals or total_calls >= budget or is_time_up():
                    return np.inf
                counter.count += 1
                val = safe_call(x)
                if val is None:
                    return np.inf
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                result = minimize(wrapped_func, x0, method='BFGS',
                                  options={'maxiter': max_evals, 'gtol': 1e-8})
            except Exception:
                pass

            # Final check
            if counter.count > 0:
                try:
                    final_val = safe_call(result.x)
                    if final_val is not None and final_val < best_f:
                        best_f = final_val
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Main search strategy
        rng = np.random.default_rng(42)

        # Phase 1: Stratified global exploration
        # Allocate ~60% of budget to global phase
        global_budget = int(budget * 0.6)
        n_lhs = min(global_budget // 2, 200)  # number of LHS points

        # Generate LHS points
        lhs_points = generate_lhs(n_lhs, dim, rng)

        # Also add some structured points: center, corners, etc.
        structured = [np.zeros(dim)]
        # Add points near the origin with small perturbations
        for scale in [0.5, 1.0, 2.0]:
            for d in range(dim):
                pt = np.zeros(dim)
                pt[d] = scale
                structured.append(pt.copy())
                pt[d] = -scale
                structured.append(pt.copy())

        # Evaluate all global points
        all_global = list(lhs_points) + structured
        # Shuffle to mix structured and random
        rng.shuffle(all_global)

        evaluated_points = []
        for pt in all_global:
            if is_time_up() or total_calls >= global_budget:
                break
            pt = np.clip(pt, -5, 5)
            f = safe_call(pt)
            if f is not None:
                evaluated_points.append((f, pt.copy()))
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

        # Sort by function value
        evaluated_points.sort(key=lambda x: x[0])

        # Phase 2: Local refinement with BFGS from best global points
        local_budget = budget - total_calls
        if local_budget > 20 and best_x is not None:
            # Take top K points for local refinement
            n_local = min(len(evaluated_points), 15)
            # Allocate local budget evenly
            per_local = max(5, local_budget // max(1, n_local))

            for i in range(n_local):
                if is_time_up() or total_calls >= budget:
                    break
                f0, x0 = evaluated_points[i]
                bfgs_local(x0, f0, per_local)

        # Phase 3: Final polish with Nelder-Mead from the best
        if not is_time_up() and total_calls < budget and best_x is not None:
            nm_budget = budget - total_calls
            if nm_budget > 10:
                # Simple Nelder-Mead implementation
                n = dim
                simplex = [best_x.copy()]
                fs = [best_f]
                step = 0.1
                for i in range(n):
                    xi = best_x.copy()
                    xi[i] += step if best_x[i] < 4.9 else -step
                    xi = np.clip(xi, -5, 5)
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    simplex.append(xi)
                    fs.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if len(simplex) == n + 1:
                    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                    max_iters = min(nm_budget, 50)
                    for _ in range(max_iters):
                        if is_time_up() or total_calls >= budget:
                            break
                        order = np.argsort(fs)
                        simplex = [simplex[i] for i in order]
                        fs = [fs[i] for i in order]

                        if abs(fs[0] - fs[-1]) < 1e-14:
                            break

                        centroid = np.mean(simplex[:-1], axis=0)
                        xr = centroid + alpha * (centroid - simplex[-1])
                        xr = np.clip(xr, -5, 5)
                        fr = safe_call(xr)
                        if fr is None:
                            break
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()

                        if fr < fs[0]:
                            xe = centroid + gamma * (centroid - simplex[-1])
                            xe = np.clip(xe, -5, 5)
                            fe = safe_call(xe)
                            if fe is None:
                                break
                            if fe < fr:
                                simplex[-1], fs[-1] = xe, fe
                                if fe < best_f:
                                    best_f = fe
                                    best_x = xe.copy()
                            else:
                                simplex[-1], fs[-1] = xr, fr
                        elif fr < fs[-2]:
                            simplex[-1], fs[-1] = xr, fr
                        else:
                            if fr < fs[-1]:
                                xc = centroid + rho * (simplex[-1] - centroid)
                            else:
                                xc = centroid - rho * (simplex[-1] - centroid)
                            xc = np.clip(xc, -5, 5)
                            fc = safe_call(xc)
                            if fc is None:
                                break
                            if fc <= fr:
                                simplex[-1], fs[-1] = xc, fc
                                if fc < best_f:
                                    best_f = fc
                                    best_x = xc.copy()
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    fi = safe_call(simplex[i])
                                    if fi is None:
                                        break
                                    fs[i] = fi
                                    if fi < best_f:
                                        best_f = fi
                                        best_x = simplex[i].copy()
                                if fi is None:
                                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
