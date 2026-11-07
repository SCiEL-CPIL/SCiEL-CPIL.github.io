"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    Uses a single, highly adaptive CMA-ES instance. CMA-ES is a state-of-the-art
    derivative-free optimizer that adapts the search distribution's covariance
    matrix to the problem's geometry. This is particularly effective for
    BBOB functions with hidden rotations and multimodal landscapes like
    Schaffers F7, as it can align the search ellipsoid with the valley directions.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # Safety margin for 330s limit

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Initial sigma: 2.0 is a reasonable start for [-5,5]^5 where min is in [-4,4]^5
        sigma = 2.0
        
        # Population size: lambda = 4 + floor(3 * log(dim + 1)) is standard, 
        # but for small dim=5, we can use a slightly larger population for better 
        # early coverage without wasting too many calls. 
        # Standard lambda for dim=5 is 4 + floor(3*log(6)) = 4 + 5 = 9.
        # Let's use lambda = 12 for slightly better exploration in the first phase.
        lam = 12
        
        # Weights: standard CMA-ES weights
        w = np.log(lam/2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mueff = int(np.floor(1/2 * lam))  # Effective population size for mean update
        
        # Adaptation parameters (standard CMA-ES)
        cmu = 1 / (3 + 0.3 * dim - mueff / 2)  # Covariance adaptation rate
        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)  # Path length adaptation for C
        cs = (mueff + 2) / (dim + mueff + 5)  # Path length adaptation for sigma
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        sigma0 = sigma
        ccovpos = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) / (dim**1.7)
        ccovneg = (1 + 2 * ((mueff - 1) / (dim + 1) - 1)) / (dim**1.7)

        # State variables
        xmean = np.zeros(dim)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        # Precompute Cholesky of C (initially I)
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)
        
        # Path vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial points: Centered start is known to be good for AOCC
        # We initialize the mean at 0, but the first generation samples around 0.
        # To ensure b_1 is decent, we could evaluate 0 first, but CMA-ES works best
        # with a population. Let's stick to standard CMA-ES initialization.
        
        x_best = None
        f_best = np.inf
        n_calls = 0
        
        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1

        # Main Loop
        while n_calls < budget and time.time() < deadline:
            # Generate population
            # X = xmean + sigma * B @ (D * Z)
            Z = rng.standard_normal((lam, dim))
            # Update Cholesky if C changed (it does every generation)
            # For dim=5, np.linalg.cholesky is fast.
            try:
                D = np.sqrt(np.diag(C))
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (rare in CMA-ES with proper regularization)
                C = np.eye(dim) * 1e-10
                D = np.sqrt(np.diag(C))
                B = np.eye(dim)
            
            # Transform
            X = xmean + sigma * (Z @ (B * D)) # (lam, dim)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            
            # Record all evaluations
            for i in range(lam):
                record(X[i], F[i])
                
                # Check budget/time mid-batch to avoid overshooting
                if n_calls >= budget or time.time() > deadline:
                    break
            
            if n_calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            xmean_new = np.sum(w[:mueff] * X_sorted[:mueff], axis=0)
            
            # Update path p_sigma
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * Binv @ (Dinv * (xmean_new - xmean) / sigma)
            # Note: Binv and Dinv are needed. 
            # Since C = B B^T, C^{-1} = (B^{-1})^T B^{-1}.
            # The standard formula uses B^{-1} C^{-1/2} or similar.
            # Simpler: y_mean = (xmean_new - xmean) / sigma
            y_mean = (xmean_new - xmean) / sigma
            
            # We need B^{-1} y_mean. 
            # Since C = B B^T, B is lower triangular.
            # Let's compute B_inv = np.linalg.inv(B)
            B_inv = np.linalg.inv(B)
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B_inv @ y_mean)
            
            # Update path p_c
            # p_c = (1 - cc) * p_c + sqrt(cc * (2 - cc) * mueff) * C^{-1/2} (xmean_new - xmean) / sigma
            # C^{-1/2} = B^{-T} D^{-1} ? No.
            # If C = B B^T, then C^{1/2} = B D? No.
            # Standard CMA-ES uses C^{-1/2} which is the inverse of the square root of C.
            # Let's use the standard update formula which often relies on the transformation.
            # Actually, the standard update is:
            # p_c = (1 - cc) p_c + sqrt(cc(2-cc)mueff) * C^{-1/2} (xmean_new - xmean)/sigma
            # C^{-1/2} can be computed as inv(cholesky(C)) * diag(1/sqrt(diag(C)))? 
            # No, if C = L L^T, C^{1/2} is not L. 
            # However, for the purpose of the direction, we can approximate or use the exact inverse sqrt.
            # For dim=5, np.linalg.inv(np.linalg.cholesky(C)) is fine.
            # Let's stick to the most robust implementation:
            # C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C)) * np.sqrt(np.diag(C)) ??? 
            # Actually, let's just use the standard formula from Hansen's paper.
            # The term is C^{-1/2} (xmean_new - xmean)/sigma.
            # Let y = (xmean_new - xmean)/sigma.
            # We want C^{-1/2} y.
            # We can compute C^{-1/2} via eigendecomposition or use the fact that 
            # if we have the Cholesky factor L of C, then C^{-1/2} is not simply related.
            # But we can compute C^{-1} easily.
            # Let's use a simpler, stable update for p_c that avoids explicit C^{-1/2} if possible, 
            # or compute it. For dim=5, eigendecomposition is cheap.
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-10, None)
            C_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
            
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * (C_inv_sqrt @ y_mean)
            
            # Update sigma
            # hsig = ||p_sigma|| / sqrt(1 - (1 - cs)^{2/gen}) / sqrt(lam) < 1.4 + 2/(lam+1)
            # gen is roughly n_calls / lam. 
            # For simplicity and stability, we use the standard threshold.
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate expected norm of p_sigma under null hypothesis
            # E[||p_sigma||] = sqrt(1 - (1 - cs)^{2/gen}) * sqrt(lam) ? 
            # Actually, the standard check is:
            # hsig = 1 if norm_p_sigma / sqrt(1 - (1 - cs)**(2 * n_evals / lam)) < 1.4 + 2 / (lam + 1)
            # Let's use a simplified version:
            gen = max(1, n_calls // lam)
            hsig = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - cs)**(2 * gen)) < 1.4 + 2 / (lam + 1) else 0.0
            
            sigma *= np.exp(cs / damps * (norm_p_sigma / np.sqrt(lam) - 1))
            
            # Update Covariance C
            # Delta_hsig = (1 - hsig) (1 + (2 - hsig) cc (1 - cc))
            delta_hsig = (1 - hsig) * (1 + (2 - hsig) * cc * (1 - cc))
            
            # Rank-1 update
            y_c = p_c / np.sqrt(cc * (2 - cc))
            C_rank1 = np.outer(y_c, y_c)
            
            # Rank-mu update
            # y_i = (X_sorted[i] - xmean) / sigma
            Y = (X_sorted[:mueff] - xmean) / sigma
            C_rankmu = np.sum(w[:mueff] * np.outer(Y, Y).transpose(0, 2, 1), axis=0)
            
            C = (1 + cmu * (mueff + 2 - 1) - cmu - ccovpos * delta_hsig) * C \
                + cmu * (C_rankmu - delta_hsig * C_rank1)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if necessary
            # Check eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += (1e-10 - np.min(eigvals)) * np.eye(dim)
            except:
                C = np.eye(dim)

            xmean = xmean_new
            
            # Restart strategy:
            # If progress is extremely slow or sigma is too small/large, restart.
            # For CMA-ES, a common restart is if the condition number is too high 
            # or if we have stagnated.
            # Given the 500 budget, we might not need complex restarts if CMA-ES is well-tuned.
            # However, Schaffers F7 is multimodal.
            # If sigma drops below 1e-4, we are likely converged to a local min.
            if sigma < 1e-4:
                # Restart from a random point or global best with larger sigma
                # Since we track global best, let's jump around it with a fresh C
                xmean = x_best + rng.standard_normal(dim) * 1.0
                xmean = np.clip(xmean, -5.0, 5.0)
                C = np.eye(dim)
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
