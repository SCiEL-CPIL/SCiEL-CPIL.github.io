"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size for f14 optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        remaining = self.budget
        
        # Directive: Population size 4
        lam = 4
        mu = max(1, lam // 2)

        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= weights.sum()
        mueff = int(np.sum(weights)**2 / np.sum(weights**2))

        # Strategy parameters (standard CMA-ES)
        c1 = (2 * (2 - mueff / (n + 2))**2) / ((n + 2)**2 + 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + 2))
        cs = (mueff + 2) / (n + mueff + 5)
        c1 += 2 * (1 - c1 - cmu) * (mueff / (n + 2) - 1)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Directive: Large initial sigma (5.0) to cover box
        xmean = np.zeros(n)
        sigma = 5.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        generation = 0

        # Initial evaluation at origin
        fx = func(xmean)
        best_x = xmean.copy()
        best_f = fx
        remaining -= 1

        # Main CMA-ES loop
        while remaining >= lam:
            generation += 1
            # Generate population
            z = rng.standard_normal((lam, n))
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                L = np.eye(n)

            x_pop = np.array([xmean + sigma * (z[i] @ L) for i in range(lam)])
            x_pop = np.clip(x_pop, -5, 5)

            # Evaluate
            f_pop = np.array([func(x_pop[i]) for i in range(lam)])
            remaining -= lam

            # Update best
            min_idx = np.argmin(f_pop)
            if f_pop[min_idx] < best_f:
                best_f = f_pop[min_idx]
                best_x = x_pop[min_idx].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Update mean using weighted recombination
            xnew = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            dx = xnew - xmean

            # Update evolution paths
            # Note: In standard CMA-ES, the update for ps involves the inverse of L if we are working in the standard normal space.
            # Here we generate x = mean + sigma * z @ L. So z = (x-mean)/(sigma) @ L^-1.
            # The path ps is updated with the mean shift in the standard normal space.
            # dx/sigma is the shift in the "L-scaled" space. We need to map it back to standard normal space?
            # Actually, standard CMA-ES update is:
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (L^T)^-1 * (xnew - xmean) / sigma
            # Let's stick to the simpler approximation often used if L is identity-like or just use the vector directly if C is close to I.
            # However, for robustness with non-identity C, we should ideally use the inverse.
            # Given the small dim (5), we can afford to compute L_inv.
            
            # Let's refine the ps update to be more accurate for non-identity C
            try:
                L_inv = np.linalg.inv(L)
                # (xnew - xmean) / sigma is in the "physical" space scaled by sigma.
                # The standard normal variable z such that x = xmean + sigma * z @ L is z = (x-xmean)/sigma @ L^-1.
                # So the mean shift in z-space is (dx/sigma) @ L^-1.
                # Wait, x = xmean + sigma * (z @ L). So z = (x-xmean)/sigma @ L^-1.
                # The vector (xnew - xmean)/sigma is a row vector? No, x is a column vector in standard math, but numpy uses row vectors in array ops often.
                # Let's treat x as 1D arrays.
                # x = xmean + sigma * (z @ L). 
                # If z is 1D (1,n), z @ L is 1D.
                # Then (x - xmean)/sigma = z @ L.
                # So z = (x - xmean)/sigma @ L^-1.
                # The update term for ps is sqrt(...) * z_mean_shift.
                z_mean_shift = (dx / sigma) @ L_inv
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean_shift
            except np.linalg.LinAlgError:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (dx / sigma)

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * generation)) / chiN) < (1.4 + 2 / (n + 1))
            
            # pc update: dx/sigma is the shift in the "L-space". 
            # Standard CMA-ES updates pc with (xnew - xmean)/sigma.
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * dx / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c1 * (2 - c1)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            
            # The rank-mu update: 
            # Sum of outer products of (x_sorted - xmean) / sigma.
            # (x_sorted - xmean) / sigma are the vectors in the "L-space".
            # Let v_i = (x_sorted[i] - xmean) / sigma.
            # The update adds cmu * sum(w_i * v_i^T v_i).
            # This updates C in the "L-space" coordinates? 
            # Actually, C represents the covariance of z @ L? No, C represents the covariance of the standard normal vector z?
            # In CMA-ES, C is the covariance matrix of the standard normal distribution z.
            # x = xmean + sigma * (z @ L). 
            # Cov(x) = sigma^2 * L^T C L.
            # The standard update updates C (covariance of z).
            # The vectors v_i = (x_i - xmean)/sigma = z_i @ L.
            # So v_i^T v_i = L^T z_i^T z_i L.
            # This would imply we are updating L^T C L, not C.
            # To update C correctly, we need to map v_i back to z_i: z_i = v_i @ L^-1.
            # Then the outer product is z_i^T z_i = L^-1 v_i^T v_i L^-T.
            
            # Let's compute the rank-mu update properly.
            try:
                # v_i are 1D arrays (n,)
                # z_i = v_i @ L_inv
                # We need sum(w_i * outer(z_i, z_i))
                # outer(z_i, z_i) is (n,n)
                
                # Efficient way:
                # Let V be (mu, n) where V[i] = (x_sorted[i] - xmean)/sigma
                V = (x_sorted[:mu] - xmean) / sigma
                # Z = V @ L_inv  (mu, n)
                Z = V @ L_inv
                # Sum of w_i * outer(Z[i], Z[i])
                # This is (Z^T * W) @ Z where W is diag(weights)
                # Or simply:
                rank_mu_update = np.sum(weights[:, None, None] * np.einsum('ij,ik->ijk', Z, Z), axis=0)
                C += cmu * rank_mu_update
            except np.linalg.LinAlgError:
                # Fallback to simpler update if L is singular (shouldn't happen with regularization)
                V = (x_sorted[:mu] - xmean) / sigma
                rank_mu_update = np.sum(weights[:, None, None] * np.einsum('ij,ik->ijk', V, V), axis=0)
                C += cmu * rank_mu_update

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 5.0)

            # Update mean
            xmean = xnew

        # Use remaining budget for local refinement around best_x
        # Use a simple (1+1)-ES style local search
        local_sigma = 0.1
        while remaining > 0:
            step = rng.standard_normal(n) * local_sigma
            y = np.clip(best_x + step, -5, 5)
            fy = func(y)
            remaining -= 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                local_sigma *= 1.5
            else:
                local_sigma *= 0.8
            local_sigma = max(1e-6, min(local_sigma, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
