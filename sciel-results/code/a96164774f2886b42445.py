# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    Uses a single adaptive CMA-ES run with careful parameter tuning
    for the 500-call budget in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        # CMA-ES implementation (lightweight, no external deps beyond numpy)
        rng = np.random.default_rng(12345)

        # Initial parameters tuned for 500 calls in dim=5
        x0 = np.zeros(dim)
        sigma0 = 2.0  # initial step size, covering most of [-5,5]^5

        # Population size: lambda = 4 + floor(3 * ln(n)) for dim=5 -> 4+3*1.6=8.8 -> 9
        # But with 500 budget, we want enough generations. 
        # lambda=6 gives ~83 generations. lambda=5 gives 100 generations.
        # Let's use lambda=6 for good balance.
        lam = 6
        mu = lam // 2  # selection size

        # Weights (log-linear)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + cs)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        mean = x0.copy()
        C = np.eye(dim)
        Cinv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chiN_val = chiN
        chiC = np.sqrt(cc * (2 - cc) * (dim + mu_eff / 2))

        def _update_eigen(C, B, D):
            """Update eigendecomposition of C."""
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip to positive
            eigvals = np.clip(eigvals, 1e-20, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            return B, D

        def _Cinv_from_BD(B, D):
            """Compute C^{-1} from eigendecomposition."""
            # C = B diag(D^2) B^T, so C^{-1} = B diag(1/D^2) B^T
            D_inv = 1.0 / D
            return B @ (D_inv[:, None] * (B.T * D_inv[None, :]))

        # Precompute
        B, D = _update_eigen(C, B, D)
        Cinv = _Cinv_from_BD(B, D)

        # Budget allocation: use ~480 calls for CMA-ES, keep 20 for safety
        max_calls = budget - 10
        n_generations = max_calls // lam

        for gen in range(n_generations):
            if self._call_count >= max_calls:
                break
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform: B @ (D * z)
                y = B @ (D * z)
                x_new = mean + sigma0 * y
                x_new = np.clip(x_new, -5.0, 5.0)
                population.append(x_new)

            # Evaluate population
            fitness = []
            for x_new in population:
                if self._call_count >= max_calls:
                    break
                if time.time() > deadline:
                    break
                fx = wrapped(x_new)
                fitness.append((fx, x_new))

            if len(fitness) < lam:
                break

            # Sort by fitness
            fitness.sort(key=lambda t: t[0])
            selected = [f[1] for f in fitness[:mu]]
            selected_f = [f[0] for f in fitness[:mu]]

            # Update mean (weighted)
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * selected[i]
            mean = mean_new

            # Update evolution paths
            # C^{1/2} = B * D (since C = B D^2 B^T)
            C_sqrt = B @ np.diag(D)
            C_sqrt_inv = B @ np.diag(1.0 / D)

            # Direction for p_c
            y_mean = C_sqrt_inv @ (mean - x0)  # Wait, should be (mean_new - mean_old)
            # Actually: y_mean = C^{-1/2} (mean_new - mean_old)
            # Let me redo: mean_old was the previous mean
            # We need to store old mean

            # Let me restructure: store old_mean before update
            # Actually, let's just use the standard update

            # Recompute properly
            # The mean update already happened. Let's use the standard CMA-ES update.

            # p_c = (1-cs) p_c + cs * sqrt(mu_eff) * C^{-1/2} (mean_new - mean_old)
            # We need mean_old. Let's track it.

            # For simplicity and robustness, let me use a simpler but effective variant:
            # Use the standard CMA-ES with proper bookkeeping.

            # Actually, let me rewrite this more carefully with proper state tracking.
            # For now, let's just do a simplified but effective version.

            break  # Exit to use the cleaner implementation below

        # Clean CMA-ES implementation
        # Reset state
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        B, D = _update_eigen(C, B, D)
        Cinv = _Cinv_from_BD(B, D)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        C_sqrt = B @ np.diag(D)
        C_sqrt_inv = B @ np.diag(1.0 / D)

        for gen in range(n_generations):
            if self._call_count >= max_calls:
                break
            if time.time() > deadline:
                break

            # Generate and evaluate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z  # B @ (D * z)
                x_new = mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                population.append(x_new)

            fitness = []
            for x_new in population:
                if self._call_count >= max_calls:
                    break
                if time.time() > deadline:
                    break
                fx = wrapped(x_new)
                fitness.append((fx, x_new))

            if len(fitness) < mu:
                break

            fitness.sort(key=lambda t: t[0])
            selected = [f[1] for f in fitness[:mu]]

            # Save old mean
            mean_old = mean.copy()

            # Update mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * selected[i]
            mean = mean_new

            # Update p_c
            y_mean = C_sqrt_inv @ (mean - mean_old)
            p_c = (1 - cs) * p_c + cs * np.sqrt(mu_eff) * y_mean

            # Update p_sigma
            # p_sigma = (1-cs) p_sigma + sqrt(cs(2-cs) mu_eff) * C_sqrt_inv @ (mean - mean_old)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ (mean - mean_old)

            # Update sigma (h-sigma rule)
            h_sigma = (
                np.linalg.norm(p_sigma) / (1 - (1 - cs) ** ((gen + 1) * cs))
                < (1.4 + 2 / (dim + 1)) * chiN
            )
            if h_sigma:
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update C
            # C = (1 - c1 - cmu) C + c1 (p_c p_c^T + (1-h_sigma) cs(2-cs) C) + cmu sum wi (yi yi^T)
            delta_h = (1 - h_sigma) * cs * (2 - cs)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + delta_h * C)
            for i in range(mu):
                yi = C_sqrt_inv @ (selected[i] - mean_old)
                C += cmu * weights[i] * np.outer(yi, yi)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update eigendecomposition
            B, D = _update_eigen(C, B, D)
            Cinv = _Cinv_from_BD(B, D)
            C_sqrt = B @ np.diag(D)
            C_sqrt_inv = B @ np.diag(1.0 / D)

            # Safety: if sigma becomes too small, break
            if sigma < 1e-10:
                break

        # Final: use remaining budget for a few random evaluations near best
        if self._best_x is not None:
            for _ in range(min(10, budget - self._call_count)):
                if self._call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                perturbation = rng.normal(0, 0.1, dim)
                x_try = np.clip(self._best_x + perturbation, -5.0, 5.0)
                wrapped(x_try)

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
