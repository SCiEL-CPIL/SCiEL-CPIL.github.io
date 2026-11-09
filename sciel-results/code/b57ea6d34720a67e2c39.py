"""CMA-ES optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        d = self.dim
        
        # Initialization
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # CMA-ES parameters
        # Default lambda is 4 + floor(3 * log(d))
        lam = int(4 + 3 * np.log(d)) if d > 1 else 2
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Initial sigma
        sigma = 2.0
        # Initial covariance
        C = np.eye(d)
        Cs = np.zeros(d)
        
        # Step sizes
        cc = (4 + mueff / d) / ((d + 4) * (2 + mueff / d))
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Eigen decomposition cache for sqrt(C)
        # We update C and compute sqrt(C) via eigendecomposition each generation
        # This is O(d^3) per generation, which is fine for d=5
        
        def sqrtm(C):
            vals, vecs = np.linalg.eigh(C)
            vals = np.sqrt(np.maximum(vals, 0))
            return vecs @ (vals[:, None] * vecs.T)
        
        B = np.eye(d)
        D = np.ones(d)
        
        for gen in range(self.budget):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lam, d))
            for i in range(lam):
                # z is standard normal
                z = rng.standard_normal(d)
                # y = B * (D * z)
                y = B @ (D * z)
                pop[i] = x + sigma * y
            
            # Evaluate
            fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Mean update
            m_new = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update path
            # y_mean = (m_new - x) / sigma
            y_mean = (m_new - x) / sigma
            Cs = (1 - cs) * Cs + cs * y_mean
            x = m_new
            
            # Update covariance matrix
            # hsig = ||Cs|| / sqrt(1 - (1-cs)^{2(gen+1)}) / sqrt(trace(C))
            # Approximate for simplicity or use standard formula
            # For d=5, we can just compute it
            hsig = 1.0  # Simplified, standard CMA-ES uses a more complex check
            # Actually, let's implement the standard hsig check roughly
            # ||Cs|| / sqrt(1 - (1-cs)^(2*(gen+1)))
            denom = np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))
            if denom > 0:
                hsig = np.linalg.norm(Cs) / denom / np.sqrt(np.trace(C))
            else:
                hsig = 1.0
            
            # Update C
            # C = (1 - mu_sigma) * C + c1 * (Cs*Cs.T + (hsig > 1 - (1-cs)^{gen+1}) * (1-cs)*cs*(1-cs)/(1-(1-cs)^{gen+1}) * C) + cmu * sum(w_i * (y_i - Cs)(y_i - Cs).T)
            # This is getting complex. Let's use a simpler rank-1 and rank-mu update without the hsig complexity if possible, 
            # or just use the standard update.
            
            # Standard update:
            # C = (1 - c1 - cmu) * C + c1 * (Cs*Cs.T + (hsig - 1) * (1 - cs) * cs * (1 - cs) / (1 - (1-cs)^(gen+1)) * C) + cmu * sum(...)
            
            # Let's simplify by just doing rank-1 and rank-mu without the hsig correction term in the c1 part for now, 
            # or use a robust implementation.
            
            # Rank-1 update
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(Cs, Cs)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (pop_sorted[i] - x) / sigma
                C += w[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Compute B and D
            vals, vecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            vals = np.maximum(vals, 1e-12)
            B = vecs
            D = np.sqrt(vals)
            
            # Update sigma (1/5 rule)
            # Approximate using the ratio of mean distance of selected to overall
            # Or use the standard CMA-ES sigma update with evolution path
            # For simplicity and robustness, let's use a simpler adaptive sigma
            # based on success rate of the last generation
            # Count how many of the top mu are better than the previous mean
            # Actually, let's just use a fixed sigma adaptation similar to (1+1) but with population
            # Better: use the standard CMA-ES sigma update with path p_sigma
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * B @ (D * y_mean)
            # sigma *= exp(cs / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            
            # Let's implement the standard sigma update
            # We need p_sigma
            if not hasattr(self, '_p_sigma'):
                self._p_sigma = np.zeros(d)
            
            # y_mean is already computed as (m_new - x_old) / sigma
            # Note: x was updated to m_new, so x_old is m_new - y_mean * sigma
            # B and D are from the PREVIOUS C, so we should have used the old B and D for y_mean calculation?
            # In standard CMA-ES, the transformation uses the current C.
            # Let's re-evaluate:
            # y = B * D * z
            # x_new = x + sigma * sum(w_i * y_i)
            # y_mean = (x_new - x) / sigma = sum(w_i * y_i)
            
            # p_sigma update
            self._p_sigma = (1 - cs) * self._p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D * y_mean))
            
            # d_sigma
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (d + 1)) + cs
            
            # E||N(0,I)||
            c_d = d - 1 + 0.5 * np.log(1 + 1.0 / d)
            E_norm = np.sqrt(d) * (1 - 1 / (4 * d) - 1 / (21 * d**2))
            
            # sigma update
            sigma *= np.exp(cs / d_sigma * (np.linalg.norm(self._p_sigma) / E_norm - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If no improvement for a long time, restart? 
            # For now, just continue.
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
