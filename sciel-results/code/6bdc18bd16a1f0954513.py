"""CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simplified CMA-ES with adaptive covariance, restarts on stagnation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # CMA-ES parameters
        lambda_pop = 10  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights for parents: linear weights normalized
        w = np.arange(1, mu + 1)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)  # effective population size

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Chi_n: expected norm of standard normal vector
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        m = np.zeros(dim)  # mean
        sigma = 2.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        B = np.eye(dim)  # eigenvectors
        D = np.ones(dim)  # eigenvalues
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)  # evolution path for C

        # Initialization: sample a few points to find a better start
        for _ in range(5):
            if total_calls >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            total_calls += 1
            if fx_try < best_f:
                best_x = x_try.copy()
                best_f = fx_try
        
        if best_x is not None:
            m = best_x.copy()
        else:
            m = np.zeros(dim)
            best_x = m.copy()
            best_f = func(m)
            total_calls += 1

        # Main loop
        while total_calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for _ in range(lambda_pop):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # y = m + sigma * B @ (D * z)
                y = m + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])
            
            # Update global best
            for y, fy in offspring:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Select parents
            parents = [y for y, fy in offspring[:mu]]
            y_w = np.sum([w[i] * parents[i] for i in range(mu)], axis=0)

            # Update mean
            m_old = m.copy()
            m = y_w

            # Update sigma evolution path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * np.linalg.norm(B @ D @ (y_w - m_old) / sigma + 1e-16)) * (B @ D @ ((y_w - m_old) / sigma))
            # Correction: The standard update is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B^T D^-1 (y_w - m_old) / sigma)
            # Let's use the standard formula properly.
            # Actually, simpler:
            # p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ D ** -1) @ ((y_w - m_old) / sigma)
            # Wait, B and D are from C = B diag(D^2) B^T.
            # Let's re-implement the update step more carefully to avoid bugs.
            
            # Re-doing the update logic for clarity and correctness
            # 1. Update m
            m = y_w
            
            # 2. Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D ** -1) @ (y_w - m_old) / sigma)
            # Note: B.T @ (D ** -1) is equivalent to (B @ diag(D))^-1 if C = B diag(D^2) B.T
            # Let's compute the inverse transform
            # The term (y_w - m_old) / sigma is the step in the standardized space.
            # We need to transform it back to the eigenbasis.
            # step_std = (y_w - m_old) / sigma
            # step_eigen = B.T @ step_std / D
            step_std = (y_w - m_old) / sigma
            step_eigen = (B.T @ step_std) / D
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step_eigen

            # 3. Update h_sigma
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls // lambda_pop + 1))) < (1 + 2 / (dim + 1)) * chi_n

            # 4. Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step_eigen

            # 5. Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (1 - h_sigma) * p_c p_c^T + c_mu * sum(w_i * (y_i - m_old)(y_i - m_old)^T / sigma^2)
            # The last term is in the eigenbasis? No, usually in the original space or transformed.
            # Standard CMA-ES updates C in the original space using the weighted outer products of the steps.
            # However, to maintain the decomposition C = B diag(D^2) B^T, we often update in the eigenbasis.
            # Let's stick to the original space update for C, then re-decompose.
            
            # Delta C in original space:
            # delta_C = c_1 * (1 - h_sigma) * np.outer(p_c, p_c)  # This is in eigenbasis, need to transform
            # Actually, p_c is in eigenbasis. So c_1 * (1-h_sigma) * B @ diag(p_c^2) @ B.T
            # And the mu-term: c_mu * sum(w_i * np.outer((y_i - m_old)/sigma, (y_i - m_old)/sigma))
            
            # Let's compute the update in the original space directly for simplicity, then re-decompose.
            # C_new = (1 - c_1 - c_mu) * C 
            # + c_1 * (1 - h_sigma) * (B @ np.diag(p_c ** 2) @ B.T)
            # + c_mu * sum(w_i * np.outer((y_i - m_old)/sigma, (y_i - m_old)/sigma))
            
            term1 = (1 - c_1 - c_mu) * C
            term2 = c_1 * (1 - h_sigma) * (B @ np.diag(p_c ** 2) @ B.T)
            
            term3 = np.zeros((dim, dim))
            for i in range(mu):
                step_i = (parents[i] - m_old) / sigma
                term3 += w[i] * np.outer(step_i, step_i)
            term3 *= c_mu

            C = term1 + term2 + term3

            # Re-decompose C = B diag(D^2) B^T
            # Ensure C is symmetric
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-10, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # 6. Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Convergence check
            if sigma < 1e-8 or np.all(D < 1e-8):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
