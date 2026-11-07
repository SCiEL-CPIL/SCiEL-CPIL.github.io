"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 10
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial state
        m = np.zeros(dim)
        sigma = 2.5
        C = np.eye(dim)
        
        # Step size adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Path vector for step size adaptation
        p_sigma = np.zeros(dim)
        
        # State tracking
        best_x = m.copy()
        best_f = np.inf
        call_count = 0
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            if call_count >= budget:
                return np.inf
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initialize population
        pop = np.zeros((lambda_, dim))
        fvals = np.zeros(lambda_)
        
        # Generate initial population
        for i in range(lambda_):
            z = np.random.randn(dim)
            pop[i] = m + sigma * z
            fvals[i] = evaluate(pop[i])
            
        idx_min = np.argmin(fvals)
        best_x = pop[idx_min].copy()
        best_f = fvals[idx_min]
        
        # Evolution loop
        while call_count < budget and time.time() < deadline:
            if sigma < 1e-12:
                break
                
            # 1. Sort population by fitness
            idx_sorted = np.argsort(fvals)
            pop_sorted = pop[idx_sorted]
            fvals_sorted = fvals[idx_sorted]
            
            # 2. Recombination
            m_old = m.copy()
            m = np.dot(weights, pop_sorted[:mu])
            
            # 3. Update Covariance Matrix
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(m - m_old, m - m_old)
            for i in range(mu):
                diff = pop_sorted[i] - m_old
                C += cmu * weights[i] * np.outer(diff, diff)
            C = (C + C.T) / 2.0
            
            # 4. Step Size Adaptation
            # Calculate Cholesky factor
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                
            # Calculate h_sigma
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(dim+1)) * E||N(0, I)||
            # E||N(0, I)|| = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            c_sigma = cc * (1 - cc)
            h_sigma_threshold = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (32.0 * dim ** 2))
            
            # Update p_sigma
            # p_sigma = (1 - cs) * p_sigma + cs * sqrt(cs * (2 - cs) * mueff) / ||B^(-1) * (m - m_old)|| * B^(-1) * (m - m_old)
            # We need B^(-1) * (m - m_old)
            # Since B is lower triangular, we can solve B * y = (m - m_old)
            try:
                # Solve B * y = (m - m_old)
                # Using np.linalg.solve for triangular is not directly supported in basic numpy without scipy, 
                # but we can use np.linalg.inv or just solve the system.
                # For dim=5, np.linalg.solve is fast enough.
                y = np.linalg.solve(B, m - m_old)
                norm_y = np.linalg.norm(y)
                if norm_y > 1e-12:
                    p_sigma = (1 - cs) * p_sigma + cs * np.sqrt(cs * (2 - cs) * mueff) / norm_y * y
                else:
                    p_sigma = (1 - cs) * p_sigma
            except np.linalg.LinAlgError:
                p_sigma = (1 - cs) * p_sigma
                
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < h_sigma_threshold else 0.0
            
            # Update sigma
            sigma *= np.exp(cs / damps * (norm_p_sigma / h_sigma_threshold - 1.0))
            
            # 5. Generate new population
            for i in range(lambda_):
                if call_count >= budget:
                    break
                # z ~ N(0, I)
                z = np.random.randn(dim)
                # x = m + sigma * B * z
                # Since B is lower triangular, B * z is efficient
                x = m + sigma * (B @ z)
                pop[i] = x
                fvals[i] = evaluate(x)
                
            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
