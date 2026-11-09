"""CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lam // 2  # recombination weight count
        # Weights (log-sum normalized)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size
        sigma = 2.0
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping for step size control
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # Initial mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, dim)
        # Covariance matrix
        C = np.eye(dim)
        # Evolution path for covariance
        pc = np.zeros(dim)
        # Evolution path for step size
        ps = np.zeros(dim)
        # Inverse square root of C (Cholesky-based)
        # Use eigenvalue decomposition for stability in low dim
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        minvD_B = (B / D) @ B.T  # C^{-1/2}
        # Initial best
        fvals = [func(m)]
        best_x = m.copy()
        best_f = fvals[0]
        call_count = 1

        # Evaluate initial population? No, start with m, then generate offspring.
        # Actually, standard CMA-ES evaluates lam offspring per generation.
        # Let's do that. But we need to be careful with budget.

        # Pre-generate first generation
        # We'll loop until budget is exhausted or time is up.

        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate lam samples
            # m + sigma * B @ z
            Z = rng.standard_normal((lam, dim))
            # Transform: X = m + sigma * (B @ Z^T)^T = m + sigma * Z @ B^T
            # Wait: C = B @ diag(D^2) @ B^T. C^{1/2} = B @ diag(D).
            # So x_i = m + sigma * B @ z_i
            X = (sigma * (Z @ B.T * D)) + m  # This is wrong. Z @ B.T * D is not B @ Z^T.
            # Correct: x_i = m + sigma * (B @ z_i)
            # In matrix form: X = m + sigma * (Z @ B.T) * D? No.
            # B is (dim, dim). z_i is (dim,). B @ z_i is (dim,).
            # Z is (lam, dim). We want each row z_i transformed by B.
            # So X = m + sigma * (Z @ B.T) * D?
            # Let's check: (Z @ B.T)_i = z_i @ B.T = (B @ z_i)^T.
            # So Z @ B.T gives rows that are (B @ z_i)^T.
            # Then multiply by D? No, D is a vector.
            # x_i = m + sigma * D * (B @ z_i)? No, D scales the eigen-directions.
            # C^{1/2} = B @ diag(D). So x_i = m + sigma * B @ diag(D) @ z_i.
            # diag(D) @ z_i = D * z_i (element-wise).
            # So x_i = m + sigma * B @ (D * z_i).
            # In matrix form: X = m + sigma * (Z * D) @ B.T
            # Because (Z * D)_i = z_i * D = diag(D) @ z_i.
            # And (Z * D) @ B.T gives rows (diag(D) @ z_i) @ B.T = (B @ diag(D) @ z_i)^T.
            # Yes.
            X = m + sigma * ((Z * D) @ B.T)
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            batch_fvals = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                f = func(X[i])
                batch_fvals.append(f)
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = X[i].copy()

            if len(batch_fvals) < lam:
                break

            fvals = np.array(batch_fvals)
            # Sort
            order = np.argsort(fvals)
            X_sorted = X[order]
            fvals_sorted = fvals[order]

            # Update mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            # C^{1/2} = B @ diag(D)
            # C^{-1/2} = B @ diag(1/D) @ B.T
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (m_new - m)/sigma
            C_inv_sqrt = (B / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (m_new - m) / sigma)

            # Update covariance
            # hsig = ||ps|| / sqrt(1-(1-cs)^{2/(gen+1)}) / sqrt(mueff)
            # Approximate hsig threshold
            hsig = 1.0  # Simplified, or compute properly
            # Proper hsig:
            # chiN = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            # We need generation count for (1-cs)^{2/gen}
            # Let's just use a fixed threshold or approximate
            hsig = 1.0 if np.linalg.norm(ps) / chiN < (1.4 + 2 / (dim + 1)) * (1 - cs) else 0.0

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - m) / sigma

            # Rank-one and rank-mu updates
            # C = (1 - mu*cmu - c1*(1+ (2-c1)*hsig)) * C + c1 * (pc*pc.T + (1-hsig)*c1*(1-cc)*C)
            # This is getting complex. Let's use a simpler update or a library-like approach.
            # Actually, let's just do a simple rank-one update and ignore rank-mu for simplicity,
            # or implement the standard formula carefully.

            # Standard CMA-ES update:
            # C = (1 - c1 - cmu) * C
            # C += c1 * (pc*pc.T + (1-hsig)*c1*(1-cc)*C)  # This term is for the rank-one correction
            # C += cmu * sum(weights_i * (y_i * y_i.T)) where y_i = (x_i - m)/sigma

            # Let's implement the standard formula:
            # C = (1 - c1 - cmu) * C
            # C += c1 * (pc*pc.T + (1-hsig)*c1*(1-cc)*C)
            # C += cmu * sum_{i=1}^{mu} weights_i * y_i * y_i.T

            # Wait, the standard formula is:
            # C = (1 - c1 - cmu) * C + c1 * (pc*pc.T + (1-hsig)*c1*(1-cc)*C) + cmu * sum(w_i * y_i y_i^T)
            # This is not quite right. Let's look it up mentally.
            # C_{t+1} = (1 - c1 - cmu) C_t + c1 (p_c p_c^T + (1 - h_sigma) c1 (1 - c_c) C_t) + cmu sum_{i=1}^mu w_i y_i y_i^T
            # where y_i = (x_i - m_t) / sigma

            # Let's implement this.
            y_sorted = (X_sorted[:mu] - m) / sigma
            rank_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)

            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * (1 - cc) * C)
            C += cmu * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))

            # Update step size
            # sigma = sigma * exp(cs/damps * (||ps||/chiN - 1))
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
