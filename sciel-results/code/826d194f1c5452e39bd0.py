"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global sampling with Latin Hypercube for better coverage
        n_initial = min(20, budget // 15)
        candidates = []
        
        # Latin Hypercube Sampling for better space coverage
        for i in range(n_initial):
            if eval_count >= budget or not time_ok():
                break
            # LHS: one point per stratum in each dimension
            x = np.array([
                rng.uniform(i / n_initial, (i + 1) / n_initial) * 10.0 - 5.0
                for _ in range(n)
            ])
            # Add small jitter to avoid exact grid points
            x += rng.uniform(-0.1, 0.1, n)
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))
        
        # Add a few pure random points for diversity
        n_random = min(5, budget - eval_count)
        for _ in range(n_random):
            if eval_count >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            # Fallback: single random point
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local refinement on top candidates
        # Allocate remaining budget across top 4 candidates (more candidates for multimodal)
        top_k = min(4, len(candidates))
        remaining = budget - eval_count
        per_candidate = max(15, remaining // max(1, top_k))

        for rank in range(top_k):
            if eval_count >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            # Nelder-Mead simplex
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Build simplex
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                delta = 0.5 if x0[j] + 0.5 <= 5 else -0.5
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            # Evaluate simplex
            fvals = [f0]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            # Nelder-Mead iterations with improved convergence handling
            max_nm_iter = per_candidate - (len(fvals) - 1)
            consecutive_no_improve = 0
            
            for _ in range(max(0, max_nm_iter)):
                if eval_count >= budget or not time_ok():
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if eval_count >= budget or not time_ok():
                    break
                fr = evaluate(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                    consecutive_no_improve = 0
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                    consecutive_no_improve = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                        consecutive_no_improve = 0
                    else:
                        # Shrink: move all points toward best
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        # Only re-evaluate the moved points (not the best)
                        for i in range(1, len(simplex)):
                            if eval_count >= budget or not time_ok():
                                break
                            fx = evaluate(simplex[i])
                            if fx is None:
                                break
                            fvals[i] = fx
                        if len(fvals) != len(simplex):
                            break
                        consecutive_no_improve += 1
                        
                        # If too many consecutive shrinks, stop this candidate
                        if consecutive_no_improve > 3:
                            break

        # Phase 3: High-precision final polish using coordinate descent + fine random search
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            
            # Coordinate descent with adaptive step
            coord_sigma = 0.05
            for _ in range(min(50, budget - eval_count)):
                if eval_count >= budget or not time_ok():
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * coord_sigma
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy is None:
                            break
                        if fy < f:
                            x, f = y, fy
                            improved = True
                        if eval_count >= budget or not time_ok():
                            break
                    if eval_count >= budget or not time_ok():
                        break
                if not improved:
                    coord_sigma *= 0.5
                    if coord_sigma < 1e-6:
                        break
            
            # Fine random search around best with very small steps
            if eval_count < budget and time_ok():
                fine_sigma = 0.02
                for _ in range(budget - eval_count):
                    if not time_ok() or eval_count >= budget:
                        break
                    y = np.clip(x + fine_sigma * rng.standard_normal(n), -5, 5)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    if fy < f:
                        x, f = y, fy
                        fine_sigma *= 1.1
                    else:
                        fine_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
