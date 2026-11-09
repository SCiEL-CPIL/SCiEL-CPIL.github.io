"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive step-size optimizer: Latin Hypercube init + Nelder-Mead with gradient-adaptive scaling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Latin Hypercube Sampling for efficient initial coverage
        # LHS provides better space-filling than a regular grid for the same number of points
        n_lhs = min(80, budget // 3)
        rng = np.random.default_rng(42)
        
        def latin_hypercube(n, d, rng):
            """Generate n points in d dimensions using Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for i in range(d):
                # Create n intervals and pick one random point from each
                intervals = rng.random(n)
                order = rng.permutation(n)
                points[order, i] = intervals
            # Map from [0,1]^d to [-4,4]^d (minimizer is in [-4,4]^5)
            points = points * 8.0 - 4.0
            return points

        lhs_points = latin_hypercube(n_lhs, dim, rng)
        
        # Evaluate LHS points
        lhs_results = []
        for idx in range(len(lhs_points)):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break
            x = lhs_points[idx]
            f = tracked_func(x)
            lhs_results.append((f, x))

        if not lhs_results:
            # Fallback: evaluate origin
            x0 = np.zeros(dim)
            f0 = tracked_func(x0)
            lhs_results = [(f0, x0)]

        # Sort and take best point for Nelder-Mead initialization
        lhs_results.sort(key=lambda t: t[0])
        center = lhs_results[0][1].copy()

        # Phase 2: Nelder-Mead with adaptive step size based on gradient magnitude
        # Estimate gradient via finite differences to adapt step size
        remaining_budget = budget - call_count
        if remaining_budget <= 0 or time.time() > deadline - 30:
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        x_min = -4.5
        x_max = 4.5
        
        # Initialize simplex: center + small perturbations
        simplex = [center.copy()]
        base_step = 0.5
        for i in range(dim):
            new_point = center.copy()
            new_point[i] += base_step
            new_point = np.clip(new_point, x_min, x_max)
            simplex.append(new_point)
        
        # Evaluate simplex
        simplex_f = [tracked_func(s) for s in simplex]
        
        # Nelder-Mead loop with adaptive step size
        max_nm_iters = min(remaining_budget // (dim + 2), 200)
        for it in range(max_nm_iters):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Sort simplex by function value
            combined = sorted(zip(simplex_f, simplex), key=lambda t: t[0])
            simplex_f = [t[0] for t in combined]
            simplex = [t[1] for t in combined]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # Check convergence: if simplex is too small, break
            centroid = np.mean(simplex[:-1], axis=0)
            spread = np.std([np.linalg.norm(s - centroid) for s in simplex], axis=0)
            if spread < 1e-8:
                break
            
            # Estimate gradient magnitude at best point to adapt step size
            # Use finite differences with a small step
            h = max(1e-4, spread * 0.01)
            grad = np.zeros(dim)
            for i in range(dim):
                xp = simplex[0].copy()
                xm = simplex[0].copy()
                xp[i] += h
                xm[i] -= h
                xp = np.clip(xp, x_min, x_max)
                xm = np.clip(xm, x_min, x_max)
                # Only evaluate if we have budget
                if call_count >= budget - 2:
                    break
                fp = tracked_func(xp)
                fm = tracked_func(xm)
                grad[i] = (fp - fm) / (2 * h)
            
            grad_norm = np.linalg.norm(grad)
            
            # Adaptive step size: large in flat regions, small near optimum
            # Scale based on inverse gradient magnitude
            if grad_norm > 1e-6:
                adaptive_step = np.clip(0.1 / grad_norm, 0.01, 2.0)
            else:
                adaptive_step = 0.01  # Very small step if gradient is nearly zero
            
            # Nelder-Mead operations
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, x_min, x_max)
            f_r = tracked_func(x_r)
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, x_min, x_max)
                f_e = tracked_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = tracked_func(x_c)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = tracked_func(x_c)
                
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], x_min, x_max)
                        simplex_f[i] = tracked_func(simplex[i])
            
            # Update best from simplex
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()
            
            # Check if we've converged
            if call_count >= budget:
                break

        # Final: use remaining budget for local refinement around best point
        remaining = budget - call_count
        if remaining > 0 and best_x is not None and time.time() < deadline - 5:
            # Adaptive local search with decreasing step size
            local_sigma = max(0.05, adaptive_step * 0.5 if 'adaptive_step' in locals() else 0.05)
            for _ in range(min(remaining, 50)):
                if call_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                y = np.clip(best_x + local_sigma * np.random.randn(dim), x_min, x_max)
                fy = tracked_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    local_sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
