"""Optimizer for BBOB f12 (Bent cigar) using quadratic surrogate + CMA-ES hybrid."""
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: quadratic surrogate for local exploitation, CMA-ES for exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # State
        x_best = np.zeros(dim)
        f_best = np.inf
        calls = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Phase 1: Quick random sampling to find a good starting region
        rng = np.random.default_rng(42)
        n_init = min(20, budget // 5)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
            if time.time() > deadline:
                return x_best, f_best

        # Phase 2: Quadratic surrogate refinement
        # Maintain a set of recent points for fitting a quadratic model
        # We'll use a simple approach: keep last K points, fit quadratic, step toward minimizer
        K = 25  # number of points to keep for quadratic fit
        points = [x_best.copy()]
        values = [f_best]

        # CMA-ES parameters
        sigma = 1.0
        # Initialize CMA-ES population
        pop_size = max(4 + int(3 * np.log(dim)), 6)
        weights = np.ones(pop_size)
        weights[:pop_size // 2] = 2.0
        weights /= weights.sum()
        mu_eff = (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (mu_eff / 2) * (c_1 * 2 - 1 / (dim + 1.5))
        c_mu = max(c_mu, 0)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES state
        mean = x_best.copy()
        C = np.eye(dim)
        step_size = np.sqrt(np.sum(np.diag(C)))
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Hessian update matrices
        H = np.zeros((dim, dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                H[i, j, i, j] = 0.5
                H[i, j, j, i] = 0.5
                H[i, j, i, j] = 0.5 if i == j else 0.5

        # Actually, let's simplify: use a standard CMA-ES with rank-1 and rank-mu updates
        # But to save time, let's use a simpler approach: combine quadratic surrogate with random restarts

        # Simpler hybrid: alternate between quadratic surrogate steps and larger random perturbations
        # This avoids the complexity of full CMA-ES while still exploiting local structure

        n_quadratic = min(150, budget - calls - 50)  # reserve some budget for final refinement
        n_random = budget - calls - n_quadratic

        # Quadratic surrogate phase
        for it in range(n_quadratic):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Build design matrix for quadratic fit
            # Use last K points + current best
            n_pts = min(K, len(points))
            if n_pts >= dim + 1:
                X = np.array(points[-n_pts:])
                Y = np.array(values[-n_pts:])

                # Fit quadratic: f(x) ≈ c0 + c^T x + x^T A x
                # Design matrix: [1, x_i, x_i_j * x_i_k for j<=k]
                n_features = 1 + dim + dim * (dim + 1) // 2
                if n_pts >= n_features:
                    # Build design matrix
                    D = np.ones((n_pts, n_features))
                    D[:, 1:1 + dim] = X
                    idx = 1 + dim
                    for i in range(dim):
                        for j in range(i, dim):
                            D[:, idx] = X[:, i] * X[:, j]
                            idx += 1

                    # Solve least squares
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(D, Y, rcond=None)
                    except:
                        coeffs = None

                    if coeffs is not None:
                        # Extract quadratic terms
                        c0 = coeffs[0]
                        c_lin = coeffs[1:1 + dim]
                        # Quadratic part
                        idx = 1 + dim
                        A = np.zeros((dim, dim))
                        for i in range(dim):
                            for j in range(i, dim):
                                val = coeffs[idx]
                                A[i, j] = val
                                if i != j:
                                    A[j, i] = val
                                idx += 1

                        # Find minimizer of quadratic: solve (2A)x = -c_lin
                        try:
                            if np.linalg.matrix_rank(2 * A) >= dim:
                                x_pred = -np.linalg.solve(2 * A, c_lin)
                                # Step toward predicted minimizer with damping
                                alpha = 0.5
                                x_new = mean + alpha * (x_pred - mean)
                                x_new = np.clip(x_new, -5, 5)
                                f_new = evaluate(x_new)
                                # Update point history
                                points.append(x_new.copy())
                                values.append(f_new)
                                if len(points) > K:
                                    points.pop(0)
                                    values.pop(0)
                                # Update mean if improved
                                if f_new < f_best:
                                    mean = x_new.copy()
                            else:
                                # Singular, take small random step
                                x_new = mean + 0.1 * rng.standard_normal(dim)
                                x_new = np.clip(x_new, -5, 5)
                                f_new = evaluate(x_new)
                                points.append(x_new.copy())
                                values.append(f_new)
                                if len(points) > K:
                                    points.pop(0)
                                    values.pop(0)
                                if f_new < f_best:
                                    mean = x_new.copy()
                        except:
                            x_new = mean + 0.1 * rng.standard_normal(dim)
                            x_new = np.clip(x_new, -5, 5)
                            f_new = evaluate(x_new)
                            points.append(x_new.copy())
                            values.append(f_new)
                            if len(points) > K:
                                points.pop(0)
                                values.pop(0)
                            if f_new < f_best:
                                mean = x_new.copy()
                    else:
                        # Fallback: random step
                        x_new = mean + 0.2 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = evaluate(x_new)
                        points.append(x_new.copy())
                        values.append(f_new)
                        if len(points) > K:
                            points.pop(0)
                            values.pop(0)
                        if f_new < f_best:
                            mean = x_new.copy()
                else:
                    # Not enough points for full quadratic, use linear or random
                    x_new = mean + 0.3 * rng.standard_normal(dim)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    points.append(x_new.copy())
                    values.append(f_new)
                    if len(points) > K:
                        points.pop(0)
                        values.pop(0)
                    if f_new < f_best:
                        mean = x_new.copy()
            else:
                # Too few points, random exploration
                x_new = mean + 0.5 * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                points.append(x_new.copy())
                values.append(f_new)
                if len(points) > K:
                    points.pop(0)
                    values.pop(0)
                if f_new < f_best:
                    mean = x_new.copy()

        # Phase 3: Final refinement with small steps
        n_final = budget - calls
        for it in range(n_final):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Use very small random steps around best
            step_size = max(0.01, 0.1 * np.sqrt(max(f_best - 0, 0)))
            x_new = x_best + step_size * rng.standard_normal(dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = evaluate(x_new)

        return x_best, f_best
