"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style population-based optimizer with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Total call tracker
        total_calls = [0]

        def tracked_func(x):
            if time.time() >= deadline:
                raise Exception("timeout")
            total_calls[0] += 1
            if total_calls[0] > budget:
                raise Exception("budget exceeded")
            return func(np.clip(x, -5, 5))

        # Initial population
        # Population size: 4 + floor(3 * ln(dim)) for dim=5 -> 4 + 3*1.609 = 4+4 = 8
        lambda_pop = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        lambda_pop = min(lambda_pop, 10)  # Cap for small budget

        # Initial mean: center of the box
        m = np.zeros(dim)
        
        # Initial sigma: roughly 1/4 of the box width
        sigma = 2.0

        # Covariance matrix: Identity
        C = np.eye(dim)
        # Cholesky factor
        L = np.eye(dim)

        # CMA-ES parameters
        mu = lambda_pop // 2  # Number of weighted points
        # Weights: log-spaced
        w = np.log(np.arange(lambda_pop, 0, -1))
        w /= w.sum()
        w = w[:mu]
        w_sum = w.sum()
        w = w / w_sum
        mueff = 1.0 / np.sum(w**2)
        
        # Step size adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        sigma_hist = []

        # Reserve budget for polish
        polish_budget = 50
        es_budget = budget - polish_budget

        # Initialize population and evaluate
        # We need to be careful with the budget here.
        # Standard CMA-ES generates lambda points per iteration.
        # If lambda_pop * iterations > es_budget, we need to stop early.
        
        # Let's just run iterations until budget is low or time is up.
        # We will generate lambda_pop points per iteration.
        
        # Pre-allocate memory for population? Not strictly necessary for small dim.
        
        while total_calls[0] < es_budget and time.time() < deadline:
            # Check if we have enough budget for a full population update
            # If not, break and go to polish
            if es_budget - total_calls[0] < lambda_pop:
                break

            # Generate population
            Y = np.empty((lambda_pop, dim))
            for i in range(lambda_pop):
                # Sample z
                z = rng.standard_normal(dim)
                # Transform using Cholesky
                # y = m + sigma * L @ z
                Y[i] = m + sigma * (L @ z)

            # Evaluate population
            F = np.empty(lambda_pop)
            for i in range(lambda_pop):
                try:
                    F[i] = tracked_func(Y[i])
                except Exception:
                    # If timeout or budget exceeded, break out
                    break
            else:
                # Success
                # Sort by fitness
                o = np.argsort(F)
                Y = Y[o]
                F = F[o]

                # Update mean
                # y_w = sum(w_i * y_i)
                y_w = np.dot(w, Y[:mu])
                
                # Update sigma (CSA)
                # ps = (1-cs)ps + sqrt(cs(2-cs)c1)/sigma * L^-1 (m_new - m_old)
                # Note: L is the Cholesky of C. L^-1 (m_new - m_old) is tricky without explicit inverse.
                # Standard CMA-ES uses: ps = (1-cs)ps + sqrt(cs(2-cs)c1) * (y_w - m) / sigma
                # But this is in the transformed space? No, usually it's:
                # ps = (1-cs)ps + sqrt(cs(2-cs)c1) * (m_new - m_old) / sigma
                # Wait, the standard formula involves the inverse of the Cholesky factor or similar.
                # Let's stick to the simpler variant where we update ps in the original space 
                # but scaled appropriately.
                
                # Actually, a robust simple implementation for small dim:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*c1) * (y_w - m) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * c1) * (y_w - m) / sigma

                # Update sigma
                sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clamp sigma
                if sigma > 50.0: sigma = 50.0
                if sigma < 1e-15: sigma = 1e-15

                # Update covariance matrix
                # hsig = (1 - (1-cs)^{2/(g+1)}) ? No, g is generation.
                # Simplified hsig: check if ps norm is reasonable
                hsig = 1.0 # Simplification: assume no saturation for now, or use threshold
                # Better: hsig = 1 if ||ps|| / sqrt(1-(1-cs)^{2g}) < (1.4 + 2/(dim+1)) * chi_n
                # Let's just use a simple threshold on ||ps||
                if np.linalg.norm(ps) > (1.4 + 2 / (dim + 1)) * chi_n:
                    hsig = 0.0

                # pc update
                pc = (1 - cmu) * pc + hsig * cs * (2 - cs) * np.outer(ps, ps)
                # Wait, standard CMA-ES pc update is:
                # pc = (1-c1)pc + hsig*sqrt(c1(2-c1)) * (m_new - m_old)/sigma
                # And C update uses pc and y_w.
                # Let's use the standard rank-one and rank-mu updates.
                
                # Recalculate pc correctly
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1)) * (y_w - m) / sigma
                
                # C update
                # C = (1 - mu1) * C + c1 * (pc^2 + (1-hsig)*c1*(1-c1)*C) + cmu * sum(wi * yi*yi^T)
                # Where yi = (y_i - m_old) / sigma
                # And mu1 = c1^2 + cmu^2 ? No.
                # Standard:
                # C = (1 - c1 - cmu) C + c1 (pc^2 + (1-hsig) c1 (1-c1) C) + cmu sum(wi (yi yi^T))
                
                # Let's compute the rank-mu part
                # Y_diff = (Y[:mu] - m) / sigma
                Y_diff = (Y[:mu] - m) / sigma
                # rank_mu = sum(wi * outer(Y_diff_i, Y_diff_i))
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(Y_diff[i], Y_diff[i])

                # Rank-one part
                rank_one = np.outer(pc, pc) + (1 - hsig) * c1 * (1 - c1) * C

                C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                # Eigenvalues check
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(dim)

                # Update Cholesky factor
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: add jitter
                    C += 1e-6 * np.eye(dim)
                    L = np.linalg.cholesky(C)

                # Update mean
                m = y_w

                # Update best
                if F[0] < best_fx:
                    best_x = Y[0].copy()
                    best_fx = F[0]
                
                # Check deadline
                if time.time() >= deadline:
                    break

            # If exception occurred in evaluation, break
            break

        # Phase 2: Nelder-Mead polish
        remaining = budget - total_calls[0]
        if remaining > 2 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(np.clip(z, -5, 5))
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
