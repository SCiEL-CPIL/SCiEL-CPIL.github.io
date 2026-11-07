"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        d = self.dim
        n = self.budget
        
        # CMA-ES Parameters
        # lambda is population size
        lam = max(4, int(4 + 3 * np.log(d)))
        
        # Initial mean: center of the box is a safe bet, or random? 
        # Random is better to avoid symmetry issues if minimum is shifted, 
        # but center is often good for BBOB. Let's use a random start to be robust.
        rng = np.random.default_rng(0)
        m = rng.uniform(-5, 5, d)
        
        # Initial sigma: large enough to explore, but not too large to waste budget
        # Typical BBOB box is [-5, 5]^d. Diagonal is ~ 10*sqrt(d).
        # Standard CMA-ES init: sigma0 = 2.0 or similar.
        sigma = 2.0
        
        # Covariance matrix C, initially identity
        C = np.eye(d)
        
        # Inverse Cholesky factor for sampling
        # We maintain C via rank-1 and rank-mu updates.
        # For simplicity and stability in small dims, we can recompute Cholesky each step or use recursive updates.
        # Given d=5, recomputing is cheap.
        
        # Step size control parameters
        cc = (4 + lam / d) / ((d + 4) + 2 * lam / d)
        cs = (2 + 2 * min(1, 1 - cc) + lam / d) / ((d + 2) + 2 * min(1, 1 - cc) + lam / d)
        cmu = min(1 - cc, 2 * (lam / 8 - 1) / ((d + 2) * (d + 2) - 1))
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1)) - 1) + cs
        
        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute weights for rank-mu update
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Main loop
        while calls < n:
            if time.time() > deadline:
                break
            
            # Sample population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite (shouldn't happen often), reset or regularize
                C = np.eye(d) + 1e-6 * np.eye(d)
                L = np.linalg.cholesky(C)
                
            # B = L, D = diag(1/sqrt(diag(B^T B)))? 
            # Standard CMA-ES sampling: y = B * D * z
            # Where B D^2 = C.
            # Let B = L (lower triangular from Cholesky C = L L^T)
            # Then D = diag(1 / sqrt(diag(L L^T))) = diag(1 / norm(L_col))
            # Actually, standard is: C = B D^2 B^T.
            # If we use Cholesky C = L L^T, we can set B = L and D = I? No.
            # Easier: sample z ~ N(0, I), then y = L @ z. Then y ~ N(0, C).
            # This is valid.
            
            z = rng.standard_normal((lam, d))
            # y_i = L @ z_i^T
            Y = (L @ z.T).T
            
            # x_i = m + sigma * y_i
            X = m[None, :] + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort indices by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Compute new mean
            m_new = m + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            
            # Update path length control variables
            # p_sigma = (1-cs) p_sigma + sqrt(cs(2-cs)damps) * B^T D^{-1} (x_new - x_old) / sigma
            # Here B D^{-1} corresponds to the inverse of the transformation used in sampling.
            # Since we sampled y = L z, the "natural" step is y/sigma.
            # The term is essentially the weighted mean of the sampled steps in the "natural" space.
            # Let's stick to the standard update on the y-vectors.
            # p_sigma = (1-cs) p_sigma + sqrt(cs(2-cs)damps) * (sum w_i y_i)
            
            y_mean = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * damps) * y_mean
            
            # h_sigma = ||p_sigma|| / sqrt(1 - (1-cs)^{2(g+1)}) / sqrt(1 - (1-cs)^{2g}) < 1.4 + 2/(d+1)
            # We approximate g (number of generations since last update) by just using the threshold
            # A common simplification: if ||p_sigma|| < (1.4 + 2/(d+1)) * sqrt(1 - (1-cs)^2) * sigma_path_len
            # Actually, let's use the standard heuristic:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^{2g}) < (1.4 + 2/(d+1)) * sqrt(1 - (1-cs)^2)
            # For simplicity and robustness in this short horizon, we can use a simpler check or fixed h_sigma=1.
            # Let's use the standard approximation:
            c_sigma_norm = np.linalg.norm(p_sigma)
            # Approximate sigma_path = sqrt(1 - (1-cs)**2) * d (rough)
            # Better: use the actual expected norm
            # E[||z||^2] = d. E[||y||^2] = tr(C) = d (init).
            # Let's just use h_sigma = 1.0 for now, it's often sufficient.
            h_sigma = 1.0
            
            # p_c = (1-cc) p_c + h_sigma * sqrt(cc(2-cc)) * (x_new - x_old) / sigma
            # In y-space:
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc)) * y_mean
            
            # Update C
            # C = (1 + cmu (2 - cmu) - cc) C + ...
            # Rank-1 update:
            # C += cc * (1 - (1-cc)^{2g}) * (p_c p_c^T - C)
            # Rank-mu update:
            # C += cmu * sum w_i (y_i - y_mean)(y_i - y_mean)^T
            
            # To avoid tracking g, we use the standard online update:
            # C = (1 - cc - cmu) C + cc * p_c p_c^T + cmu * sum w_i (y_i - y_mean)(y_i - y_mean)^T
            # Wait, the standard update is:
            # C <- (1 + mu_eff (2 - mu_eff) - cc) C + cc p_c p_c^T + mu_eff sum w_i (y_i - y_mean)(y_i - y_mean)^T
            # Note: mu_eff is used for the rank-mu part.
            
            H = p_c[:, None] * p_c[None, :]
            H += np.sum(w[:, None, None] * Y_sorted[:mu, :, None] * Y_sorted[:mu, None, :], axis=0)
            
            C = (1 + mueff * (2 - mueff) - cc) * C + cc * H + mueff * np.sum(w[:, None, None] * (Y_sorted[:mu, :, None] - y_mean[None, :, None]) * (Y_sorted[:mu, None, :] - y_mean[None, :]), axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-6:
                C += (1e-6 - np.min(eigvals)) * np.eye(d)
                
            # Update sigma
            # sigma *= exp(cs / damps * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(2/d) * Gamma((d+1)/2) / Gamma(d/2)
            # For d=5, this is approx 2.128
            # Let's compute it properly
            from math import gamma
            if d == 1:
                c_sigma = 1.0
            else:
                c_sigma = np.sqrt(2.0 / d) * gamma((d + 1) / 2.0) / gamma(d / 2.0)
                
            sigma *= np.exp((cs / damps) * (c_sigma_norm / c_sigma - 1.0))
            
            # Clamp sigma to prevent explosion or collapse
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            m = m_new
            m = np.clip(m, -5, 5)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
