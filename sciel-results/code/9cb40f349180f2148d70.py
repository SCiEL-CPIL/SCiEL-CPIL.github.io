"""BBOB f22 optimizer: Adaptive Hyperparameter Annealing with single population."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Hyperparameter Annealing optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0
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
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Initial diverse points using LHS
        n_lhs = 10
        lhs_points = np.zeros((n_lhs, n))
        for i in range(n):
            perm = rng.permutation(n_lhs)
            u = (perm + 0.5) / n_lhs
            lhs_points[:, i] = -5 + 10 * u

        # Evaluate initial points
        candidates = []
        for i in range(n_lhs):
            if not check_time() or calls_made >= self.budget:
                break
            fx = safe_func(lhs_points[i])
            candidates.append((fx, lhs_points[i].copy()))

        if not candidates:
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Main loop: Adaptive population-based search
        # Parameters
        pop_size = 10
        mu = 5  # number of best candidates to use for recombination
        sigma = 1.0  # initial step size
        c_sigma = 0.5  # adaptation rate for sigma
        c_cov = 0.1  # adaptation rate for covariance
        cov = np.eye(n)

        # Track improvement rate for adaptive annealing
        best_f_history = [best_f]
        no_improve_count = 0
        improvement_threshold = 1e-6

        for iteration in range(500):
            if not check_time() or calls_made >= self.budget:
                break

            # Generate new candidates from best mu points
            new_candidates = []
            for j in range(pop_size):
                if not check_time() or calls_made >= self.budget:
                    break

                # Select parent indices (weighted by rank)
                parent_indices = rng.choice(mu, size=2, replace=False)
                p1 = candidates[parent_indices[0]][1]
                p2 = candidates[parent_indices[1]][1]

                # Recombination
                alpha = rng.uniform(0, 1)
                x_child = alpha * p1 + (1 - alpha) * p2

                # Mutation
                x_child = x_child + sigma * (cov @ rng.standard_normal(n))
                x_child = np.clip(x_child, -5, 5)

                fx = safe_func(x_child)
                new_candidates.append((fx, x_child.copy()))

            if not new_candidates:
                break

            # Merge with existing candidates and keep best pop_size
            all_candidates = candidates + new_candidates
            all_candidates.sort(key=lambda t: t[0])
            candidates = all_candidates[:pop_size]

            # Update best
            if candidates[0][0] < best_f:
                best_f = candidates[0][0]
                best_x = candidates[0][1].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive sigma annealing based on improvement rate
            if no_improve_count > 5:
                sigma *= 0.9  # slow down
            elif no_improve_count == 0:
                sigma = min(sigma * 1.1, 5.0)  # speed up if improving

            # Update covariance based on recent best points
            if len(candidates) >= mu:
                recent_best = np.array([c[1] for c in candidates[:mu]])
                mean_recent = np.mean(recent_best, axis=0)
                centered = recent_best - mean_recent
                cov_new = np.eye(n) * 0.9 + 0.1 * (centered.T @ centered) / mu
                cov = (cov + cov_new) / 2
                # Normalize
                trC = np.trace(cov) / n
                cov = cov / max(trC, 1e-10)

            # Convergence check
            if candidates[0][0] - candidates[-1][0] < 1e-10:
                break

            # Early stopping if sigma is very small
            if sigma < 1e-6:
                break

        # Final polish with Nelder-Mead
        if check_time() and calls_made < self.budget and best_f < np.inf:
            self._nelder_mead(func, best_x, best_f, safe_func, check_time, rng)

        return best_x, best_f

    def _nelder_mead(self, func, x_start, f_start, safe_func, check_time, rng):
        nonlocal_calls = {'calls_made': 0, 'best_x': x_start.copy(), 'best_f': f_start}
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        simplex = [x_start.copy()]
        simplex_f = [f_start]
        for i in range(dim):
            xi = x_start.copy()
            xi[i] += 0.2
            xi = np.clip(xi, -5, 5)
            if np.allclose(xi, x_start):
                xi[i] = x_start[i] + 0.02
                xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            if not check_time():
                return
            fx = safe_func(simplex[i])
            simplex_f.append(fx)

        for step in range(30):
            if not check_time():
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = safe_func(xr)
            if not check_time():
                break

            if fxr < simplex_f[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = safe_func(xe)
                if not check_time():
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
                if not check_time():
                    break
                if fxc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if not check_time():
                            return
                        simplex_f[i] = safe_func(simplex[i])

            if np.max(simplex_f) - np.min(simplex_f) < 1e-13:
                break

# EVOLVE-BLOCK-END
