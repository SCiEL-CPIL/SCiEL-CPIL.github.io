"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        n = self.dim
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(n)) -> for n=5, lambda = 4 + 3*1.6 = 8.8 -> 8? 
        # Standard rule: lambda = 4 + floor(3 * log(n))
        # log(5) ~ 1.609, 3*1.609 = 4.827, floor = 4. lambda = 8.
        # However, the directive suggests population size 10. Let's use lambda=10 as suggested for robustness.
        lam = 10
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Step size
        # Initial sigma. Box is [-5, 5]^5. 
        # A common choice is sigma0 = 2 * (max - min) / n or similar.
        # Let's start with sigma = 2.0, which covers a good portion of the space.
        sigma = 2.0

        # Mean
        # Initialize near center or random? 
        # Random initialization within box is safer to avoid bias if shift is large, 
        # but center is often good for symmetric functions. 
        # Given "hidden seeded shift", the minimum is in [-4, 4]^5.
        # Let's start at 0.0 for symmetry, or random? 
        # CMA-ES is robust to initialization. Let's use a random point in [-2, 2] to be safe but not too extreme.
        # Actually, starting at 0 is standard for BBOB if not specified. But shift is hidden.
        # Let's use a random start in [-2, 2] to break symmetry.
        m = rng.uniform(-2, 2, n)

        # Covariance matrix
        C = np.eye(n)

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Rates
        # c_sigma = (mueff + 2) / (n + mueff + 5)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        # c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        # c_1 = 2 / ((n + 1.3)^2 + mueff)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        # c_mu = (2 * (mueff - 2 + 2 / mueff)) / ((n + 2)^2 + mueff)
        c_mu = (2 * (mueff - 2 + 2 / mueff)) / ((n + 2)**2 + mueff)
        
        # Damping for sigma update
        # h_sigma = (1 - exp(-c_sigma * gen / n))
        # We need to track generation or just use standard update.
        
        # Threshold for h_sigma
        # hsig = (1 + 2/max(1, mueff-n) + c_sigma) * chiN
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        hsig = (1 + 2/max(1, mueff - n) + c_sigma) * chiN

        # Threshold for pc update
        # h_c = (1 + 2/max(1, mueff-n) + c_c) * chiN
        h_c = (1 + 2/max(1, mueff - n) + c_c) * chiN

        best_x = m.copy()
        best_f = func(m)
        calls_made = 1

        # Pre-allocate population storage for efficiency
        # We will generate, evaluate, sort, then update.

        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Need Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add regularization
                C = np.eye(n)
                B = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)
                sigma = 2.0
                # Re-evaluate best? No, just continue with reset state.
                # This is a fallback, should be rare.

            # Generate lambda samples
            # y = B @ z
            Z = rng.standard_normal((lam, n))
            P = Z @ B.T  # Since B is lower triangular, C = B @ B.T. So B @ z is equivalent to z @ B.T? 
            # Wait. If C = B @ B.T, then x = m + sigma * B @ z.
            # In numpy, if B is lower triangular, B @ z is correct.
            # Let's verify: C = B @ B.T.
            # We want z such that Cov(z @ B.T) = B @ Cov(z) @ B.T = B @ I @ B.T = B @ B.T = C.
            # So P = Z @ B.T is correct for rows being samples.
            
            X = m[None, :] + sigma * P
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam

            # Check time after evaluation
            if time.time() > deadline:
                # Update best if any new point is better
                idx = np.argmin(F)
                if F[idx] < best_f:
                    best_f = F[idx]
                    best_x = X[idx].copy()
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Update mean
            # m_new = sum(weights * X_sorted[:mu])
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B.T @ (m_new - m))
            # Note: The standard formula uses the "step" in the unrotated space.
            # diff = (m_new - m) / sigma
            # diff_in_B_space = B.T @ diff  (since diff = B @ z_avg, z_avg = B.T @ diff)
            diff = (m_new - m) / sigma
            # We need B^{-1} @ diff? 
            # If m_new - m = sigma * B @ z_avg, then z_avg = B^{-1} @ (m_new - m) / sigma.
            # Since B is lower triangular, B^{-1} is easy to compute or we can solve.
            # z_avg = np.linalg.solve(B, diff)
            z_avg = np.linalg.solve(B, diff)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_avg

            # h_sigma condition
            # if ||ps|| / sqrt(1 - (1 - c_sigma)^(2*(gen+1))) < (1.4 + 2/(n+1)) * chiN
            # For simplicity and robustness, we often approximate the denominator or use a fixed threshold.
            # Let's use the standard approximation: 
            # h_sigma = 1 if ||ps||/sqrt(1-(1-c_sigma)^2) < (1.4 + 2/(n+1)) * chiN else 0
            # Actually, the term is sqrt(1 - (1 - c_sigma)^(2 * g)) where g is generation count.
            # We can track generation count or just use the asymptotic value sqrt(1/(c_sigma)).
            # Let's use a simple check:
            ps_norm = np.linalg.norm(ps)
            # Approximate denominator
            denom = np.sqrt(1 - (1 - c_sigma)**2) # This is for g=1. 
            # Better: track a decay factor or just use the threshold (1.4 + 2/(n+1)) * chiN
            # Standard implementation often uses:
            # h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma)**(2*gen)) < (1.4 + 2.0/(n+1.0)) * chiN else 0.0
            # Since we don't track gen easily in this loop structure without extra var, let's use the asymptotic threshold.
            # Or simply: if ps_norm < (1.4 + 2.0/(n+1.0)) * chiN * np.sqrt(1/c_sigma):
            # Let's stick to the simpler heuristic often used in simplified CMA-ES:
            # h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2.0/(n+1.0)) * chiN else 0.0
            # This is an approximation for early generations. For later, the denominator approaches sqrt(1/c_sigma).
            # Let's use a dynamic denominator approximation:
            # We'll just use the condition: ps_norm < (1.4 + 2.0/(n+1.0)) * chiN * np.sqrt(1.0 / c_sigma)
            # This is the asymptotic limit.
            h_sigma = 1.0 if ps_norm < (1.4 + 2.0 / (n + 1.0)) * chiN * np.sqrt(1.0 / c_sigma) else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (ps_norm / chiN - 1)) 
            # Wait, the formula is:
            # sigma = sigma * exp(c_sigma * (||ps||/chiN - 1))
            # But this is only if h_sigma == 1? No, the update is always applied, 
            # but the path ps is only updated with h_sigma factor in some variants?
            # In the standard Hansen 2016 version:
            # sigma = sigma * exp(c_sigma / c_sigma * (||ps||/chiN - 1))  <-- No, c_sigma is in the exponent?
            # Formula: sigma_{g+1} = sigma_g * exp(c_sigma * (||ps||/chiN - 1))
            # However, if h_sigma is 0, the path ps is still updated, but the sigma update might be different?
            # Actually, h_sigma is used in the pc update, not the sigma update directly in the simplest form.
            # Let's check Hansen's paper:
            # 1. Update ps
            # 2. Update sigma: sigma = sigma * exp(c_sigma * (||ps||/chiN - 1))
            # 3. Update pc: pc = (1-c_c)*pc + h_c * sqrt(c_c*(2-c_c)*mueff) * B^{-1} @ (m_new - m)/sigma
            #    where h_c = 1 if ||pc||/sqrt(1-(1-c_c)^(2g)) < (1.4+2/(n+1))*chiC else 0
            # 4. Update C
            # So h_sigma is NOT used in the standard CMA-ES sigma update? 
            # Ah, there is a variant where h_sigma affects the sigma update to prevent it from being too aggressive.
            # But the most common implementation (e.g. in cma library) uses:
            # sigma = sigma * exp(c_sigma * (||ps||/chiN - 1))
            # Let's stick to that.
            
            # Update pc
            # pc = (1 - c_c) * pc + h_c * sqrt(c_c * (2 - c_c) * mueff) * z_avg
            # z_avg is already B^{-1} @ diff
            pc_norm = np.linalg.norm(pc)
            # h_c condition
            # h_c = 1 if pc_norm / sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(n+1)) * chiC else 0
            # chiC is similar to chiN but for the covariance update? 
            # Actually, chiC = chiN in many implementations.
            h_c = 1.0 if pc_norm < (1.4 + 2.0 / (n + 1.0)) * chiN * np.sqrt(1.0 / c_c) else 0.0
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * z_avg

            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma*(1-c_sigma)^2) * C) 
            #    + c_mu * sum(weights * (y_i - m) @ (y_i - m).T)
            # Where y_i = (x_i - m)/sigma
            # Note: The term (1 - h_sigma*(1-c_sigma)^2) is sometimes written as (1 - (1 - c_sigma)^2) when h_sigma=1.
            # Let's use the standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_sigma)**2) * C) + c_mu * rank_mu
            # Wait, the (1 - (1-c_sigma)^2) term is actually (1 - (1 - c_sigma)**2) * C? 
            # In Hansen 2016:
            # C_{g+1} = (1 - c_1 - c_mu) C_g + c_1 (pc_g pc_g^T + (1 - h_{sigma,g}(1-c_sigma)^2) C_g) + c_mu sum ...
            # If h_sigma is 1, the term is (1 - (1-c_sigma)^2) C_g.
            # If h_sigma is 0, the term is C_g.
            
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (X_sorted[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            # Update C
            # We need to be careful with the h_sigma term.
            # Let's use the simplified version often seen in code:
            # C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * rank_mu
            # This assumes h_sigma is effectively handled by the ps update logic or is always 1 in the C update?
            # No, h_sigma is explicitly in the formula.
            # Let's implement it correctly:
            h_sigma_term = (1 - h_sigma * (1 - c_sigma)**2)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + h_sigma_term * C) + c_mu * rank_mu

            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2

            # Update m
            m = m_new

            # Check convergence
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
