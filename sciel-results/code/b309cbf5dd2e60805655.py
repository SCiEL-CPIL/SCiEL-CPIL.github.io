"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s wall-clock

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For dim=5, lambda (population size) is typically 4 + floor(3*ln(n))
        lam = 4 + int(3 * np.log(dim))  # for dim=5, ln(5)~1.6, 3*1.6=4.8, floor=4, so lam=8
        lam = max(lam, 4)

        # Initial step size
        sigma = 1.0

        # Initial center
        rng = np.random.default_rng(42)
        x = np.zeros(dim)  # start at origin, Rastrigin global min is at 0 before rotation
        # Actually for rotated Rastrigin, the minimum is shifted. Let's start at 0.
        
        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cs = (2 + 2 / (dim + 1. / 3.)) / (dim + 2 + 2 + 2 / (dim + 1. / 3.))
        cmu = min(0.3, (2 + 2 / (dim + 1. / 3.)) / ((dim + 2) ** 2 - 2))
        cmu = min(cmu, (1 - cc) / 2)
        damps = (1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cs)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        # Precompute
        # We need to generate lam candidates per iteration
        # Budget: 500 evaluations total. 
        # Each iteration uses lam evaluations.
        # 500 / 8 = 62.5 iterations.

        # Evaluation counter
        n_evals = 0

        # Initial evaluation
        fx = func(x)
        n_evals += 1
        best_x = x.copy()
        best_f = fx

        # CMA-ES loop
        while n_evals < budget:
            if time.time() > deadline:
                break

            # Generate candidates
            # We need to update C and sigma based on previous step if this is not the first
            # Actually, standard CMA-ES:
            # 1. Sample z_i ~ N(0, I)
            # 2. y_i = B z_i
            # 3. x_i = m + sigma y_i
            # 4. Evaluate f(x_i)
            # 5. Sort
            # 6. Update m, sigma, C, pc, ps

            # For simplicity and robustness, let's implement a simplified (mu+lambda)-CMA-ES
            # or just use the standard update.
            
            # Sample z
            Z = rng.standard_normal((lam, dim))
            
            # Transform to Y
            # B = C^{1/2}
            # Use eigen decomposition for C^{1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals))
            
            Y = Z @ B.T
            
            # Candidates
            X = x[None, :] + sigma * Y
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                n_evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            if n_evals >= budget or time.time() > deadline:
                break
                
            # If we broke early in evaluation, we can't do a full update step properly
            # But let's assume we completed the batch.
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update weights
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)
            
            # Update mean
            x_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (w @ Z[order[:mu]].T).T)
            
            # Update covariance
            if np.linalg.norm(ps) < (1.4 + 1 / cs) * chi_n:
                C = (1 - cmu) * C + cmu / damps * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * C)
            
            # Update covariance with rank-mu update
            diff = X_sorted[:mu] - x[None, :]
            C = (1 - cmu * (1 - (1 - cs) ** 2)) * C + cmu * np.sum(w[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            # Clip sigma
            sigma = np.clip(sigma, 1e-12, 1e3)
            
            # Update mean
            x = x_new
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
