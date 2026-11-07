"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coordinate descent with adaptive step sizes for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Multi-scale deterministic coordinate descent
        # Start from origin, use decreasing step sizes, prioritize axes by recent gain
        x = np.zeros(dim)
        f = eval_point(x)
        best_x = x.copy()
        best_f = f

        # Multi-scale steps: start coarse, refine fine
        steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

        for step_size in steps:
            if calls >= budget or time_left() < 1:
                break

            # Track per-axis gains to prioritize
            axis_gains = np.zeros(dim)

            improved = False
            # Multiple passes to allow convergence within this scale
            max_passes = 3
            for _pass in range(max_passes):
                if calls >= budget or time_left() < 1:
                    break

                # Sort axes by descending recent gain
                axis_order = np.argsort(-axis_gains)

                for d in axis_order:
                    if calls >= budget or time_left() < 1:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[d] += step_size
                    x_plus = np.clip(x_plus, -5, 5)
                    if calls < budget and time_left() > 0.5:
                        f_plus = eval_point(x_plus)
                        if f_plus < f:
                            gain = f - f_plus
                            axis_gains[d] += gain
                            x = x_plus
                            f = f_plus
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[d] -= step_size
                    x_minus = np.clip(x_minus, -5, 5)
                    if calls < budget and time_left() > 0.5:
                        f_minus = eval_point(x_minus)
                        if f_minus < f:
                            gain = f - f_minus
                            axis_gains[d] += gain
                            x = x_minus
                            f = f_minus
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()

                if not improved:
                    break

            if not improved and step_size < 0.1:
                break  # No improvement with small steps, we've converged

        # Phase 2: Multi-start refinement from best point and nearby points
        # Generate candidate starts around the best point found
        start_candidates = [best_x.copy()]

        # Add points at various distances from best (closer for refinement)
        for dist in [1.0, 0.5, 0.25]:
            for d in range(dim):
                for s in [-1, 1]:
                    pt = best_x.copy()
                    pt[d] += s * dist
                    pt = np.clip(pt, -5, 5)
                    start_candidates.append(pt)

        # Add diagonal neighbors for better coverage in 5D
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        pt = best_x.copy()
                        pt[d1] += s1 * 0.5
                        pt[d2] += s2 * 0.5
                        pt = np.clip(pt, -5, 5)
                        start_candidates.append(pt)

        # Add some deterministic spread points at corners
        for d in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = s * 4.5
                start_candidates.append(pt)

        # Evaluate and sort starts
        start_fvals = []
        for pt in start_candidates:
            if calls >= budget or time_left() < 5:
                break
            # Check if already evaluated (approximate)
            already = False
            for sp, sf in start_fvals:
                if np.allclose(sp, pt, atol=1e-8):
                    already = True
                    break
            if not already:
                fp = eval_point(pt)
                start_fvals.append((pt.copy(), fp))
                if fp < best_f:
                    best_f = fp
                    best_x = pt.copy()

        start_fvals.sort(key=lambda x: x[1])

        # Phase 3: Nelder-Mead from top starts
        # Allocate more budget to Nelder-Mead for better convergence
        K = min(6, len(start_fvals), max(1, (budget - calls) // 30))

        for i in range(K):
            if calls >= budget - 5 or time_left() < 5:
                break

            x0 = start_fvals[i][0].copy()
            f0 = start_fvals[i][1]

            # Nelder-Mead simplex method
            simplex = [x0.copy()]
            simplex_f = [f0]

            # Add dim additional points
            for d in range(dim):
                new_pt = x0.copy()
                new_pt[d] += 0.3
                new_pt = np.clip(new_pt, -5, 5)
                if calls < budget and time_left() > 1:
                    new_f = eval_point(new_pt)
                    simplex.append(new_pt)
                    simplex_f.append(new_f)
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_pt.copy()
                else:
                    break

            if len(simplex) < 2:
                continue

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Adaptive iteration count: more iterations for earlier (better) starts
            remaining = budget - calls
            max_iter = min(80, max(15, remaining // (K - i)))

            for _ in range(max_iter):
                if calls >= budget - 5 or time_left() < 2:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[j].copy() for j in order]
                simplex_f = [simplex_f[j] for j in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)

                if calls >= budget or time_left() < 1:
                    break
                fr = eval_point(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 1:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        break
                    fe = eval_point(xe)
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)

                    if calls >= budget or time_left() < 1:
                        break
                    fc = eval_point(xc)
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink: move all points toward best
                        for j in range(1, len(simplex)):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            if calls >= budget or time_left() < 1:
                                break
                            simplex_f[j] = eval_point(simplex[j])
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
