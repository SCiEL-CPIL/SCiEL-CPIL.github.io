"""Optimizer using a two-phase strategy for BBOB f13 (Sharp Ridge).

Phase 1: Active sampling to identify the ridge direction and curvature.
Phase 2: Constrained line-search descent orthogonal to the ridge, then fine-tuning along the ridge.
Phase 3: (1+1)-ES polish for final convergence.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Two-phase optimizer for Sharp Ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Active sampling to identify ridge structure ----------
        # Use a small number of evaluations to sample the function and fit a low-rank quadratic model
        n_samples = 30
        samples_x = []
        samples_f = []

        # Initial random sampling
        for _ in range(n_samples):
            if evaluated >= budget or not elapsed_ok():
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            f = eval_point(x)
            samples_x.append(x)
            samples_f.append(f)

        # Add structured points around the best found
        if best_x is not None:
            for _ in range(10):
                if evaluated >= budget or not elapsed_ok():
                    break
                x = clip(best_x + rng.normal(0, 0.5, size=dim))
                f = eval_point(x)
                samples_x.append(x)
                samples_f.append(f)

        # Fit a low-rank quadratic model to identify ridge direction
        # f(x) ≈ c + (x - x0)^T B (x - x0)
        # For a sharp ridge, B has one large eigenvalue (ridge direction) and small eigenvalues (cross-ridge)
        if len(samples_x) >= dim + 1:
            X = np.array(samples_x)
            f = np.array(samples_f)
            x0 = X.mean(axis=0)
            X_centered = X - x0

            # Fit quadratic model: f(x) = c + x^T B x
            # Using least squares
            # f_i = c + x_i^T B x_i
            # Let z_i = [1, x_i[0], ..., x_i[dim-1], x_i[0]^2, ..., x_i[dim-1]^2, x_i[0]*x_i[1], ...]
            # But for low-rank, we can just use the centered points
            # f_i - f_mean ≈ (x_i - x0)^T B (x_i - x0)
            f_mean = f.mean()
            f_centered = f - f_mean

            # Solve for B using least squares
            # f_centered_i ≈ sum_{j,k} B[j,k] * X_centered[i,j] * X_centered[i,k]
            # This is a linear system in the entries of B
            # Let's build the design matrix
            A = np.zeros((len(X), dim * dim))
            for i in range(len(X)):
                for j in range(dim):
                    for k in range(dim):
                        A[i, j * dim + k] = X_centered[i, j] * X_centered[i, k]
            b_vec = f_centered

            # Solve least squares
            try:
                B_flat, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                B = B_flat.reshape(dim, dim)
                B = (B + B.T) / 2  # Symmetrize
            except np.linalg.LinAlgError:
                B = np.eye(dim)

            # Find ridge direction: eigenvector corresponding to largest eigenvalue
            eigvals, eigvecs = np.linalg.eigh(B)
            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigvals))[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Ridge direction is the eigenvector with the largest eigenvalue
            ridge_dir = eigvecs[:, 0]
            # Ensure consistent sign
            if ridge_dir[0] < 0:
                ridge_dir = -ridge_dir

            # Cross-ridge subspace: orthogonal complement of ridge_dir
            # Build basis for orthogonal complement
            # Use QR decomposition
            Q, _ = np.linalg.qr(np.column_stack([ridge_dir, np.eye(dim)]))
            cross_basis = Q[:, 1:]  # dim x (dim-1)
        else:
            # Fallback: random ridge direction
            ridge_dir = rng.normal(size=dim)
            ridge_dir /= np.linalg.norm(ridge_dir)
            Q, _ = np.linalg.qr(np.column_stack([ridge_dir, np.eye(dim)]))
            cross_basis = Q[:, 1:]

        # ---------- Phase 2: Constrained descent ----------
        # First, optimize along the cross-ridge directions to get to the ridge
        # Then, optimize along the ridge direction
        if best_x is not None and evaluated < budget and elapsed_ok():
            x = best_x.copy()
            fx = best_f

            # Step 2a: Cross-ridge optimization using Nelder-Mead in the orthogonal subspace
            # Project x onto the ridge line: x_ridge = x - (x - x0) * (ridge_dir^T (x - x0)) / ||ridge_dir||^2
            # Actually, let's just do a local search in the cross-ridge subspace
            # Use a simple coordinate descent in the cross-ridge basis

            # Find the point on the ridge closest to x
            # x = x_ridge + x_cross
            # x_cross = cross_basis @ (cross_basis^T (x - x0))
            x0 = np.zeros(dim)  # Assume ridge passes through origin for simplicity
            x_cross_coeff = cross_basis.T @ (x - x0)
            x_ridge_coeff = ridge_dir.T @ (x - x0)

            # Optimize x_cross_coeff to minimize f
            # Use a simple (1+1)-ES in the cross-ridge subspace
            sigma_cross = 0.5
            successes = 0
            window = 0
            WIN = 5

            for _ in range(min(50, budget - evaluated)):
                if not elapsed_ok() or evaluated >= budget:
                    break
                # Perturb in cross-ridge subspace
                delta_cross = sigma_cross * rng.standard_normal(dim - 1)
                new_x_cross_coeff = x_cross_coeff + delta_cross
                new_x = x0 + ridge_dir * x_ridge_coeff + cross_basis @ new_x_cross_coeff
                new_x = clip(new_x)
                new_f = eval_point(new_x)
                window += 1
                if new_f < fx:
                    x, fx = new_x, new_f
                    x_cross_coeff = new_x_cross_coeff
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    successes += 1
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma_cross *= 1.3
                    else:
                        sigma_cross *= 0.7
                    sigma_cross = float(np.clip(sigma_cross, 1e-6, 2.0))
                    successes = 0
                    window = 0

            # Step 2b: Optimize along the ridge direction
            # Use a line search along the ridge
            sigma_ridge = 0.5
            successes = 0
            window = 0
            WIN = 5

            for _ in range(min(50, budget - evaluated)):
                if not elapsed_ok() or evaluated >= budget:
                    break
                # Perturb along the ridge
                delta_ridge = sigma_ridge * rng.standard_normal()
                new_x_ridge_coeff = x_ridge_coeff + delta_ridge
                new_x = x0 + ridge_dir * new_x_ridge_coeff + cross_basis @ x_cross_coeff
                new_x = clip(new_x)
                new_f = eval_point(new_x)
                window += 1
                if new_f < fx:
                    x, fx = new_x, new_f
                    x_ridge_coeff = new_x_ridge_coeff
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    successes += 1
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma_ridge *= 1.3
                    else:
                        sigma_ridge *= 0.7
                    sigma_ridge = float(np.clip(sigma_ridge, 1e-6, 2.0))
                    successes = 0
                    window = 0

        # ---------- Phase 3: (1+1)-ES polish ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            successes = 0
            window = 0
            WIN = 5

            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    successes += 1
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-8, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
