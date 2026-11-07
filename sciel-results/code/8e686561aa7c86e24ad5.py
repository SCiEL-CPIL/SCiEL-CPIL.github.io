"""Optimizer for BBOB f8 (Rosenbrock) with CMA-ES and NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for Rosenbrock minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Wrap function to track calls and time
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped(x):
            call_count[0] += 1
            fx = float(func(x))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # CMA-ES parameters
        dim = self.dim
        budget = self.budget

        # Initial population: mix of random and structured points
        rng = np.random.default_rng(42)

        # Start near origin (Rosenbrock minimum is at 0 in standard form)
        # but with some spread to explore
        x_center = np.zeros(dim)
        sigma = 1.5

        # CMA-ES parameters
        lambda_ = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim) * sigma
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def sample_points(center, sigma, B, D, n):
            """Generate n candidate points using CMA-ES distribution."""
            zs = rng.standard_normal((n, dim))
            # Transform to covariance
            points = center[None, :] + sigma * (zs @ (B * D[None, :]).T)
            return np.clip(points, -5, 5)

        # Initial evaluation
        x0 = np.clip(x_center + sigma * rng.standard_normal(dim), -5, 5)
        f0 = wrapped(x0)

        # Evaluate initial population
        population = sample_points(x0, sigma, B, D, lambda_)
        fvals = np.array([wrapped(p) for p in population])

        # Sort by fitness
        indices = np.argsort(fvals)
        population = population[indices]
        fvals = fvals[indices]

        # Weighted mean of best mu
        w_mean = np.average(population[:mu], axis=0, weights=w)

        # Update evolution paths
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D.T @ (B.T @ (w_mean - x0) / sigma)
        norm_ps = np.linalg.norm(ps) / chi_n
        if norm_ps < (1.4 + 1 / dim) * np.sqrt(1 - (1 - c_sigma) ** 2) / (1 - c_sigma):
            pc = (1 - c_c) * pc + c_c / c_sigma * (w_mean - x0) / sigma

        # Update covariance matrix
        hsig = (1.04 + 2 / (dim + 1)) * (1 - (1 - c_sigma) ** (budget - call_count[0]))
        delta_sig = (1 - (1 - c_sigma) ** (budget - call_count[0])) / (1 - (1 - c_sigma) ** (2 * (budget - call_count[0])))
        rank_mu = np.outer(w_mean - x0, w_mean - x0)
        rank_one = np.outer(pc, pc)
        D2 = D ** 2
        D_inv = 1.0 / D
        B_inv = B.T  # B is orthogonal

        # Update D using BFGS-like low-rank update
        # Simplified: use eigenvalue-based update
        C = (D_inv[:, None] * B_inv) @ (
            (1 - c_1 - c_mu) * np.eye(dim)
            + c_1 * (rank_one + hsig * delta_sig * np.outer(pc, pc))
            + c_mu * (D_inv[:, None] * rank_mu)
        ) @ (D_inv[None, :] * B)

        # Symmetrize
        C = (C + C.T) / 2

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        D = np.sqrt(eigenvalues)
        B = eigenvectors

        # Update center and sigma
        x0 = w_mean
        sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
        sigma = np.clip(sigma, 1e-12, 10.0)

        # Continue CMA-ES iterations
        for iteration in range(1, budget):
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Sample new population
            population = sample_points(x0, sigma, B, D, lambda_)
            fvals = np.array([wrapped(p) for p in population])

            # Sort
            indices = np.argsort(fvals)
            population = population[indices]
            fvals = fvals[indices]

            # Weighted mean
            w_mean = np.average(population[:mu], axis=0, weights=w)

            # Update paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D.T @ (B.T @ (w_mean - x0) / sigma)
            norm_ps = np.linalg.norm(ps) / chi_n
            if norm_ps < (1.4 + 1 / dim) * np.sqrt(1 - (1 - c_sigma) ** 2) / (1 - c_sigma):
                pc = (1 - c_c) * pc + c_c / c_sigma * (w_mean - x0) / sigma

            # Update covariance
            hsig = (1.04 + 2 / (dim + 1)) * (1 - (1 - c_sigma) ** (budget - call_count[0]))
            delta_sig = (1 - (1 - c_sigma) ** (budget - call_count[0])) / (1 - (1 - c_sigma) ** (2 * (budget - call_count[0])))
            rank_mu = np.outer(w_mean - x0, w_mean - x0)
            rank_one = np.outer(pc, pc)
            D2 = D ** 2
            D_inv = 1.0 / D
            B_inv = B.T

            C = (D_inv[:, None] * B_inv) @ (
                (1 - c_1 - c_mu) * np.eye(dim)
                + c_1 * (rank_one + hsig * delta_sig * np.outer(pc, pc))
                + c_mu * (D_inv[:, None] * rank_mu)
            ) @ (D_inv[None, :] * B)
            C = (C + C.T) / 2

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D = np.sqrt(eigenvalues)
            B = eigenvectors

            x0 = w_mean
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Check for convergence
            if np.linalg.norm(D) < 1e-10 or sigma < 1e-12:
                break

        # Nelder-Mead polish on the best point found
        if best_x[0] is not None and call_count[0] < budget:
            x_nm = best_x[0].copy()
            f_nm = best_f[0]

            # Initialize simplex
            simplex = [x_nm.copy()]
            f_simplex = [f_nm]
            for i in range(dim):
                xi = x_nm.copy()
                step = 0.1 if x_nm[i] != 0 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = wrapped(xi)
                simplex.append(xi)
                f_simplex.append(fi)

            # Nelder-Mead iterations
            for _ in range(min(100, budget - call_count[0])):
                if call_count[0] >= budget or time.time() > deadline - 3:
                    break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = wrapped(xr)

                if fr < f_simplex[-2]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = wrapped(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[0]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = wrapped(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = wrapped(simplex[i])

                # Update best
                if f_simplex[0] < best_f[0]:
                    best_f[0] = f_simplex[0]
                    best_x[0] = simplex[0].copy()

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
