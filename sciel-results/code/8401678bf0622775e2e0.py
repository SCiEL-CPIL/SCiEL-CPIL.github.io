"""Two-phase adaptive optimizer for BBOB f22: broad exploration then intensive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

class Optimizer:
    """Hybrid surrogate-guided optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = func(x)
            except Exception:
                return np.inf
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else np.inf

        rng = np.random.default_rng(42)

        def nelder_mead(x_start, f_start, max_steps=60, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                # Convergence check: if simplex is very tight, stop
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < 1e-12:
                    break
                # Also stop if positions are very close
                pos_spread = np.max(np.abs(simplex - simplex[0]))
                if pos_spread < 1e-8:
                    break

        def coo_local_search(x0, f0, max_iter=40):
            """Coordinate descent local search."""
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = f0
            step = 0.5
            for it in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break
                improved = False
                for i in range(self.dim):
                    if not check_time() or calls_made >= self.budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return x, f

        # Phase 1: Broad exploration - larger LHS + structured random to cover 21-peak landscape
        n_lhs = 50
        points = []
        
        # Latin Hypercube Sampling
        if calls_made + n_lhs <= self.budget:
            lhs_points = np.zeros((n_lhs, n))
            for i in range(n):
                perm = rng.permutation(n_lhs)
                u = (perm + 0.5) / n_lhs
                lhs_points[:, i] = -5 + 10 * u
            for x in lhs_points:
                if not check_time() or calls_made >= self.budget:
                    break
                fx = safe_func(x)
                points.append((fx, x.copy()))

        # Add structured random points in [-4, 4]^5 where minima lie
        for _ in range(10):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-4, 4, size=n)
            fx = safe_func(x)
            points.append((fx, x.copy()))

        # Center point
        if check_time() and calls_made < self.budget:
            fx = safe_func(np.zeros(n))
            points.append((fx, np.zeros(n)))

        points.sort(key=lambda t: t[0])

        # Phase 2: Intensive local refinement on top 2 candidates
        # Use ~30% of remaining budget for first, rest for second
        top_k = min(2, len(points))
        remaining = self.budget - calls_made
        
        for i in range(top_k):
            if not check_time() or calls_made >= self.budget:
                break
            x_start = points[i][1].copy()
            f_start = points[i][0]
            
            # First local search: moderate delta, more steps
            delta = 0.6 if i == 0 else 0.4
            max_steps = int(remaining * 0.45) if i == 0 else int(remaining * 0.35)
            max_steps = max(30, max_steps)
            nelder_mead(x_start, f_start, max_steps=max_steps, initial_delta=delta)
            
            # Coordinate descent polish after each local search
            if check_time() and calls_made < self.budget:
                coo_local_search(best_x.copy(), best_f, max_iter=15)
            
            remaining = self.budget - calls_made

        # Phase 3: Final intensive polish from the global best
        if check_time() and calls_made < self.budget and best_f < np.inf:
            # Fine Nelder-Mead with small initial delta
            nelder_mead(best_x.copy(), best_f, max_steps=40, initial_delta=0.15)
            
            # Final coordinate descent
            if check_time() and calls_made < self.budget:
                coo_local_search(best_x.copy(), best_f, max_iter=15)
            
            # One more very fine NM
            if check_time() and calls_made < self.budget:
                nelder_mead(best_x.copy(), best_f, max_steps=20, initial_delta=0.05)

        return best_x, best_f

# EVOLVE-BLOCK-END
