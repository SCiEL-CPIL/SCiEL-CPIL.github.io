"""Two-point secant initialization + CMA-ES for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-point secant corner search followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Two-point secant corner search to find the optimal vertex
        # For a linear function f(x) = c^T x + b, the minimum on [-5,5]^d
        # is at the vertex x_i = -5*sign(c_i) for each i.
        # We can determine sign(c_i) by comparing f at two points differing only in dimension i.
        # Use x0 = -5*ones and x_i = x0 with x0[i] flipped to +5.
        # If f(x_i) < f(x0), then c_i > 0, so optimal x_i = -5.
        # If f(x_i) > f(x0), then c_i < 0, so optimal x_i = +5.

        x0 = np.full(dim, -5.0)
        f0 = func(x0)
        calls = 1

        # Determine the sign of each coefficient
        signs = np.zeros(dim)  # signs[i] = sign of c_i
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = x0.copy()
            xi[i] = 5.0
            fi = func(xi)
            calls += 1
            if fi < f0:
                signs[i] = 1.0  # c_i > 0, so optimal is at -5
            else:
                signs[i] = -1.0  # c_i < 0, so optimal is at +5

        # Construct the optimal vertex
        x_vertex = np.full(dim, -5.0)
        x_vertex[signs < 0] = 5.0
        f_vertex = func(x_vertex)
        calls += 1

        # For a purely linear function, this vertex IS the global minimum.
        # But due to the hidden rotation, the function isn't simply linear in the
        # original coordinates - it's linear after a rotation. So the corner
        # search in original coordinates won't directly find the minimum.
        # However, it gives us a good starting point and gradient direction info.

        # Actually, let me reconsider. The function is f(x) = c^T R(x - x*) + f_opt
        # where R is a rotation. This is still linear: f(x) = (R^T c)^T x - (R^T c)^T x* + f_opt
        # So it IS linear in x, just with a rotated coefficient vector.
        # The corner search WILL find the exact minimum vertex!

        # But wait - the problem says "Linear slope" which for BBOB f5 means:
        # f(x) = c^T x where c has components from [1, 10] (or similar)
        # After rotation and shift, it's still linear.
        # So the vertex solution is exact for a linear function on a box.

        # However, there might be numerical issues or the function might not be
        # perfectly linear. Let's use CMA-ES as a refinement step.

        # Use the vertex as starting point for CMA-ES
        x_best = x_vertex.copy()
        f_best = f_vertex

        # If we have budget remaining, run CMA-ES from the vertex
        remaining = budget - calls
        if remaining > 10:
            # CMA-ES parameters
            sigma = 0.5  # start with small sigma since we're near optimum
            popsize = max(4, int(4 + 3 * np.log(dim)))
            # Adaptive parameters
            c_sigma = (popsize + 2) / (dim + popsize + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((popsize - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + popsize / dim) / (dim + 4 + 2 * popsize / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + popsize)
            c_mu = min(1 - c_1, 2 * ((popsize / 2 + 1) / (dim + 2) ** 2 - 1))

            # Covariance matrix
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initial population
            for iteration in range(remaining):
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for j in range(popsize):
                    z = rng.standard_normal(dim)
                    y = x_best + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append((y, func(y)))
                    calls += 1
                    if calls >= budget:
                        break

                if len(offspring) < popsize:
                    break

                # Sort by fitness
                offspring.sort(key=lambda x: x[1])

                # Update best
                if offspring[0][1] < f_best:
                    x_best = offspring[0][0].copy()
                    f_best = offspring[0][1]

                # Selection weights
                mu = popsize // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights ** 2)

                # Mean of selected individuals
                x_new = np.zeros(dim)
                for k in range(mu):
                    x_new += weights[k] * offspring[k][0]

                # Update evolution paths
                if np.linalg.norm(x_new - x_best) > 1e-16:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (x_new - x_best) / sigma)
                    h_sigma = min(1.0, np.linalg.norm(p_sigma) / (1 - (1 - c_sigma) ** (iteration + 1)) / chi_n)
                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_best)
                else:
                    h_sigma = 1.0

                # Update covariance matrix
                delta_d = (1 + h_sigma * (c_c - 1) - c_c) if h_sigma < 1.0 else (1 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for k in range(mu):
                    v = C_sqrt.T @ (offspring[k][0] - x_best) / sigma
                    rank_mu += weights[k] * np.outer(v, v)

                C = (1 + c_c * (delta_d - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update C_sqrt via Cholesky (with regularization if needed)
                for _ in range(10):
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                        break
                    except np.linalg.LinAlgError:
                        C = C + 1e-3 * np.eye(dim)
                else:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)

                x_best = x_new

        return x_best, f_best
# EVOLVE-BLOCK-END
