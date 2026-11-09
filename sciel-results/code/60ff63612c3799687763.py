"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search followed by CMA-ES-like refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Track best solution found
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x, f

        # Phase 1: Coarse random search to find basin of attraction
        # Use a portion of the budget for initial exploration
        n_phase1 = min(100, self.budget // 5)
        rng = np.random.default_rng(42)

        # Start with a few strategic points
        # Center of the box
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        update_best(x0, f0)

        # Random points spread across the box
        for i in range(n_phase1 - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            update_best(x, f)

        # Phase 2: CMA-ES-like local refinement from best point
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        # CMA-ES parameters
        dim = self.dim
        mu = max(1, dim // 2)  # number of offspring
        sigma = 1.0  # initial step size
        c1 = (2) / ((dim + 1)**2 + 5)  # learning rate for covariance
        cmu = min(1 - c1, 2 * (mu - 2 + 1 / mu) / ((dim + 2)**2 + mu))  # learning rate for covariance
        cc = (4 + dim / mu) / (dim + 4 + 2 * mu / dim)  # path length for sigma
        cs = (mu + 2 + cc) / (dim + mu + 5)  # path length for sigma
        hsig = (1.7 + 2 / (dim + 1)) * (1 - cc)  # threshold for sigma adaptation
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Adaptive step size based on phase 1 results
        # Estimate local scale from the spread of good solutions
        # For now, start with a reasonable sigma
        sigma = 0.5

        remaining_budget = self.budget - (n_phase1 if best_x is not None else 0)
        # Adjust for any calls made in phase 1
        calls_made = n_phase1

        for gen in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate mu offspring
            offspring = []
            for i in range(mu):
                if time.time() > deadline:
                    break
                # Sample from multivariate normal with covariance C
                z = rng.standard_normal(dim)
                y = best_x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[0])

            # Weighted recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            x_new = sum(w * y for w, (_, y) in zip(weights, offspring))

            # Update paths
            D_inv = 1.0 / D
            C_inv_sqrt = np.linalg.inv(B) * D_inv
            # Simplified: use diagonal approximation for speed
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_new - best_x) / sigma
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * hsig

            # Update covariance
            C = (1 - c1 - cmu) * C
            delta_h = (1 - hs) * 2 - cc
            pc = (1 - cc) * pc + delta_h * np.sqrt(cc * (2 - cc) * mueff) * C_inv_sqrt @ (x_new - best_x) / sigma
            C += c1 * np.outer(pc, pc)
            for w, (_, y) in zip(weights, offspring[:mu]):
                v = C_inv_sqrt @ (y - best_x) / sigma
                C += cmu * w * np.outer(v, v)

            # Update sigma
            sigma *= np.exp(cs / sigma * (np.linalg.norm(ps) / hsig - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update best
            if offspring[0][0] < best_f:
                best_x, best_f = offspring[0][1], offspring[0][0]

            # Check convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
