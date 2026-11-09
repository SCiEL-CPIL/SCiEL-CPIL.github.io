import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_f = np.inf
        best_x = None
        calls_made = 0

        # Phase 1: Deterministic grid-search warm-up to bracket the minimum
        # Use a coarse grid with strategic points to locate the basin quickly
        # This gives early progress for AOCC scoring
        
        # Start with center
        x_center = np.zeros(dim)
        fx_center = func(x_center)
        calls_made += 1
        best_x = x_center.copy()
        best_f = fx_center

        # Evaluate axis-aligned points at various distances
        axis_points = []
        for i in range(dim):
            for d in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = d
                axis_points.append(p)

        # Evaluate diagonal points
        diag_points = []
        for sign in [-1, 1]:
            for d in [2.0, 4.0]:
                p = np.full(dim, sign * d)
                diag_points.append(p)

        # Combine and evaluate all warm-up points
        warmup_points = axis_points + diag_points
        for p in warmup_points:
            if time.time() > deadline or calls_made >= budget:
                break
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: CMA-ES refinement
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        best_x, best_f, calls_made = self._cma_es_refine(
            func, best_x, best_f, remaining, dim, rng, deadline
        )

        return best_x, best_f

    def _cma_es_refine(self, func, best_x, best_f, budget, dim, rng, deadline):
        """CMA-ES with restart logic for stagnation."""
        lam = 12  # Population size
        mu = lam // 2

        # Initial sigma: moderate, will adapt
        sigma = 1.5
        mean = best_x.copy()

        # CMA-ES parameters
        weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).sum()
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = self._eigh_inv_sqrt(C)

        calls_made = 0
        no_improve_count = 0
        stagnation_threshold = 20

        for gen in range(budget):
            if time.time() > deadline or calls_made >= budget:
                break

            # Generate candidates
            z_samples = rng.standard_normal((lam, dim))
            candidates = np.array([mean + sigma * (C_inv_sqrt @ z_samples[j]) for j in range(lam)])
            candidates = np.clip(candidates, -5, 5)

            # Evaluate
            fvals = []
            for x_j in candidates:
                if time.time() > deadline or calls_made >= budget:
                    fvals.append(best_f)
                    continue
                f_j = func(x_j)
                calls_made += 1
                fvals.append(f_j)
                if f_j < best_f:
                    best_f = f_j
                    best_x = x_j.copy()

            fvals = np.array(fvals)

            # Check improvement
            gen_best = np.min(fvals)
            if gen_best < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Sort and select
            order = np.argsort(fvals)
            x_selected = candidates[order[:mu]]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_selected, axis=0)

            # Update paths and covariance
            if np.linalg.norm(mean - mean_old) > 1e-10:
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (mean - mean_old))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (mean - mean_old)

                delta_hsig = 1.0 if hsig else 0.0
                C = (1 + c1 * delta_hsig + cmu) * C - c1 * np.outer(p_c, p_c)
                diff = (C_inv_sqrt @ (x_selected - mean_old).T).T
                C -= cmu * np.sum(weights[:, None, None] * np.outer(diff, diff), axis=0)

                C_inv_sqrt = self._eigh_inv_sqrt(C)

                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(gen + 1)) - 1))
                sigma = np.clip(sigma, 1e-6, 8.0)

            # Stagnation restart
            if no_improve_count >= stagnation_threshold:
                mean = best_x.copy()
                sigma = max(sigma * 1.5, 0.5)
                C = np.eye(dim)
                C_inv_sqrt = self._eigh_inv_sqrt(C)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0

        return best_x, best_f, calls_made

    def _eigh_inv_sqrt(self, C):
        """Eigen-decompose C, clip eigenvalues, return C_inv_sqrt."""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-20, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return C_inv_sqrt
