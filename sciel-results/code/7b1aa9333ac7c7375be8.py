"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Halton initialization followed by a single CMA-ES-like (1+1) run."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        # Deterministic Halton sequence for space-filling initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate Halton points in [0,1)^dim
        num_halton = 30
        halton_points = []
        for i in range(1, num_halton + 1):
            point = np.array([halton(i, p) for p in [2, 3, 5, 7, 11][:dim]])
            # Map to [-5, 5]
            point = point * 10.0 - 5.0
            halton_points.append(point)

        # Add a few strategic points: origin, corners
        halton_points.append(np.zeros(dim))
        halton_points.append(np.full(dim, 2.0))
        halton_points.append(np.full(dim, -2.0))

        # Evaluate Halton points to find initial best
        best_x = None
        best_f = np.inf
        calls = 0

        for pt in halton_points:
            if calls >= budget or time.time() > deadline:
                break
            x = np.clip(pt, -5, 5)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: single random point
            rng = np.random.default_rng(0)
            x = rng.uniform(-5, 5, dim)
            fx = float(func(x))
            calls += 1
            best_x = x.copy()
            best_f = fx

        # Now run a (1+1)-ES with CMA-ES-like covariance adaptation
        # Initialize at the best Halton point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # CMA-ES parameters
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation rates for dim=5
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 + dim / 3.0) / (dim + 12)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((2 + dim / 3.0) ** 2 - dim) / ((dim + 2) ** 2 + dim))

        rng = np.random.default_rng(42)

        # Success rate tracking for sigma adaptation (1/5 rule)
        successes = 0
        total = 0
        success_window = 10

        while calls < budget and time.time() < deadline:
            # Generate candidate using current covariance
            y = x + sigma * (C_sqrt @ rng.standard_normal(dim))
            y = np.clip(y, -5, 5)
            fy = float(func(y))
            calls += 1

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Adaptation
            if fy < fx:
                # Success
                successes += 1
                total += 1
                x = y
                fx = fy

                # Update sigma using 1/5 rule approximation
                sigma *= 1.08
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

                # Update covariance
                d = (y - x) / sigma
                C = (1 - c_1) * C + c_1 * (np.outer(d, d) - C)
                C = (C + C.T) / 2

                # Update path variables
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * c_1) * d
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * c_1) * d
            else:
                # Failure
                total += 1
                sigma *= 0.92
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

                # Update path variables (no step)
                p_sigma = (1 - c_sigma) * p_sigma
                p_c = (1 - c_c) * p_c

            # Periodically recompute square root of covariance
            if calls % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)

            # Reset success window periodically
            if total >= success_window:
                success_rate = successes / total
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.3:
                    sigma *= 0.8
                successes = 0
                total = 0

            if sigma < 1e-7:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
