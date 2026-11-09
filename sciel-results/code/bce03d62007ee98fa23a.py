import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 10: lam = 10
        # mu: number of parents
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Step size
        sigma = 2.0
        # Mean
        m = np.zeros(dim)
        # Covariance
        C = np.eye(dim)
        # Inverse sqrt of C for sampling
        # We need to sample from N(0, C). We can use Cholesky or Eigendecomposition.
        # For simplicity and speed in small dim, Eigendecomposition is fine.
        
        # Learning rates
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        cmu = min(1 - cc, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cs
        
        # Chi lambda
        chis = np.sqrt(dim * (1 - 1/(4*dim) - 1/(20*dim**2)))
        
        # Iterations
        max_iter = budget // lam
        
        best_f = np.inf
        best_x = m.copy()
        
        def counting_func(x):
            nonlocal self, best_f, best_x
            self._call_count += 1
            if self._call_count > budget:
                return 1e10 # Return large value to stop, but we check count outside
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        ps = np.zeros(dim) # Evolution path for step size
        pc = np.zeros(dim) # Evolution path for covariance
        
        for k in range(max_iter):
            if self._call_count >= budget:
                break
                
            # Sample
            # Get eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            # Sample from N(0, I)
            ys = np.random.randn(lam, dim)
            # Transform to N(0, C)
            # y = eigvecs * diag(sqrt(eigvals)) * z
            # Actually, we want samples from N(0, C).
            # C = V D V^T. Sqrt(C) = V D^0.5 V^T.
            # Sample = Sqrt(C) * z
            sqrt_eigvals = np.sqrt(eigvals)
            # y_i = sum_j V_ij sqrt(D_jj) z_ij ... wait.
            # y = V @ diag(sqrt_eigvals) @ z.T ?
            # Let's just use Cholesky for stability if possible, or just random rotation.
            # For dim=5, Eigh is fast.
            # y = (ys * sqrt_eigvals) @ eigvecs.T
            
            ys = (ys * sqrt_eigvals) @ eigvecs.T
            
            xs = np.array([m + sigma * y for y in ys])
            
            # Evaluate
            fs = np.array([counting_func(x) for x in xs])
            
            # Sort
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update evolution path for step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma) * (m - m_old)
            
            # Update step size
            # hsig = 1 if ||ps||/sqrt(1-(1-cs)^{2(k+1)}) < (1.4 + 2/(dim+1)) * chis else 0
            # Approximation for hsig:
            ps_norm = np.linalg.norm(ps)
            hsig = 1.0 # Simplified, or compute properly
            # Proper hsig:
            # k is iteration index
            # threshold = (1.4 + 2/(dim+1)) * chis
            # denom = np.sqrt(1 - (1-cs)**(2*(k+1)))
            # if ps_norm/denom < threshold: hsig = 1 else 0
            # Let's use the simplified version or just skip hsig if it's complex.
            # Actually, hsig is important for stability.
            # Let's implement it.
            threshold = (1.4 + 2/(dim+1)) * chis
            # We need the cumulative factor.
            # For simplicity, let's just use hsig=1 if ps_norm < 1.5*chis? No.
            # Let's stick to a robust simplified CMA-ES that might be slightly less optimal but stable.
            
            sigma *= np.exp(cs / damps * (ps_norm / chis - 1))
            
            # Update evolution path for covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (m - m_old)
            
            # Update covariance
            # C = (1 - cmu - cc) * C + cc * pc * pc.T + cmu * sum w_i y_i y_i.T
            # Note: y_i are the normalized steps (before scaling by sigma).
            # The formula is:
            # C <- (1 - cmu - cc) C + cc pc pc^T + cmu sum_{i=1}^mu w_i y_i y_i^T
            # Wait, the standard update is:
            # C <- (1 - cmu - cc) C + cc pc pc^T + cmu * (1/sigma^2) * sum w_i (x_i - m_old) (x_i - m_old)^T
            # Which is equivalent to using y_i.
            
            # We need to be careful with the terms.
            # Let's use the standard update.
            # y_i are the sampled vectors.
            # The term is sum w_i y_i y_i^T
            rank_mu = np.sum(w[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
            
            C = (1 - cmu - cc) * C + cc * np.outer(pc, pc) + cmu * rank_mu
            
            # Ensure C is symmetric positive definite
            # Symmetrize
            C = (C + C.T) / 2
            # Clip eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
        return best_x, best_f
