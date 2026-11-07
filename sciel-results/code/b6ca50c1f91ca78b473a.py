"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: axis-aligned binary search + CMA-ES refinement for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget_remaining, deadline, rng, best_x, best_f, calls):
        """Run CMA-ES from given initial state, returning updated best and call count."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        c_sigma = (mu + 2) / (dim + 4 + mu)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        p_sigma = np.zeros(dim)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        p_c = np.zeros(dim)

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        no_improve = 0

        for gen in range(300):
            if time.time() > deadline or calls >= budget_remaining:
                break

            population = []
            for i in range(lam):
                if calls >= budget_remaining or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if len(population) < mu:
                break

            population.sort(key=lambda t: t[0])
            selected = [y for _, y in population[:mu]]
            y_mean = np.sum([w * y for w, y in zip(weights, selected)], axis=0)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu) * C_sqrt_inv @ (y_mean - x) / sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) - 1 / (25 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu) * C_sqrt_inv @ (y_mean - x) / sigma

            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * (mu_eff - 1) + c_mu * (mu_eff - 1) - c_1 - c_mu + c_1 * delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum([w * np.outer(y - x, y - x) for w, y in zip(weights, selected)], axis=0) / sigma ** 2

            x = y_mean

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            if sigma < 1e-10 or (no_improve > 80 and sigma < 0.05):
                break

        return best_x, best_f, calls

    def _axis_binary_search(self, func, x, dim, budget, deadline, calls, best_x, best_f, n_steps=4):
        """For each axis, perform a coarse binary search to find the step boundary."""
        for axis in range(dim):
            if calls >= budget or time.time() > deadline:
                break

            # Try moving in positive and negative directions to find which side has the step
            step_size = 3.0
            best_val = best_f
            best_pos = x.copy()

            # Evaluate at current position
            for direction in [1, -1]:
                if calls >= budget or time.time() > deadline:
                    break

                # Binary search along this axis
                lo = x[axis] - direction * step_size
                hi = x[axis] + direction * step_size
                lo = max(lo, -5.0)
                hi = min(hi, 5.0)
                if lo > hi:
                    lo, hi = hi, lo

                # Sample a few points to find where the function changes
                samples = np.linspace(lo, hi, n_steps + 1)
                prev_val = None
                for s in samples:
                    if calls >= budget or time.time() > deadline:
                        break
                    test_x = x.copy()
                    test_x[axis] = s
                    test_x = np.clip(test_x, -5, 5)
                    fv = func(test_x)
                    calls += 1

                    if fv < best_val:
                        best_val = fv
                        best_pos = test_x.copy()

                    if prev_val is not None:
                        # If we see a significant drop, we may have found the step
                        if prev_val - fv > 1e-6:
                            # Refine the boundary between prev sample and current
                            boundary_lo = samples[samples.tolist().index(s) - 1]
                            boundary_hi = s
                            # A few refinement steps
                            for _ in range(3):
                                mid = (boundary_lo + boundary_hi) / 2
                                if calls >= budget or time.time() > deadline:
                                    break
                                test_x = x.copy()
                                test_x[axis] = mid
                                test_x = np.clip(test_x, -5, 5)
                                fv_mid = func(test_x)
                                calls += 1
                                if fv_mid < fv:
                                    boundary_hi = mid
                                    best_pos = test_x.copy()
                                    if fv_mid < best_val:
                                        best_val = fv_mid
                                else:
                                    boundary_lo = mid
                            # Set to the better side
                            test_x = x.copy()
                            test_x[axis] = boundary_hi
                            test_x = np.clip(test_x, -5, 5)
                            fv_final = func(test_x)
                            calls += 1
                            if fv_final < best_val:
                                best_val = fv_final
                                best_pos = test_x.copy()
                            break

                    prev_val = fv

            if best_val < best_f:
                best_f = best_val
                best_x = best_pos.copy()
                x = best_pos.copy()

        return x, best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration with a small grid
        # Evaluate at center and a few strategic points
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Try a few random starting points
        for _ in range(min(10, budget - 1)):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, dim)
            fv = func(x_try)
            calls += 1
            if fv < best_f:
                best_f = fv
                best_x = x_try.copy()

        # Phase 2: Axis-aligned binary search to find step boundaries
        x = best_x.copy()
        x, best_x, best_f, calls = self._axis_binary_search(
            func, x, dim, budget, deadline, calls, best_x, best_f, n_steps=5
        )

        # Phase 3: Second pass of axis search from improved position
        if calls < budget and time.time() < deadline:
            x, best_x, best_f, calls = self._axis_binary_search(
                func, x, dim, budget, deadline, calls, best_x, best_f, n_steps=4
            )

        # Phase 4: CMA-ES refinement
        if calls < budget and time.time() < deadline:
            sigma_init = 0.5
            best_x, best_f, calls = self._cma_es(
                func, best_x, sigma_init, budget, deadline, rng, best_x, best_f, calls
            )

        # Phase 5: Final local refinement with small steps
        if calls < budget and time.time() < deadline:
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                for axis in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for step in [0.1, 0.05, 0.01, 0.005]:
                        for direction in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            test_x = best_x.copy()
                            test_x[axis] += direction * step
                            test_x = np.clip(test_x, -5, 5)
                            fv = func(test_x)
                            calls += 1
                            if fv < best_f:
                                best_f = fv
                                best_x = test_x.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
