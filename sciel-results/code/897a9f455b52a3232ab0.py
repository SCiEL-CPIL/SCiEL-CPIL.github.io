import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 - mueff))
        
        # State
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n) # Cholesky factor of C
        D = 1.0       # sqrt(det(C))
        
        ps = np.zeros(n)
        ph = np.zeros(n)
        
        best_x = mean.copy()
        best_f = func(best_x)
        eval_count = 1
        
        hsig = (1 + 2.0 / (n + 1)) * np.sqrt((n + 1) * (n - 1) + 1)
        
        while eval_count < budget:
            if time.time() > time_limit:
                break
            
            # Generate population
            Z = np.random.standard_normal((lam, n))
            # y = mean + sigma * B @ Z.T
            Y = mean + sigma * (B @ Z.T).T
            Y = np.clip(Y, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            eval_count += lam
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Update mean
            # y_w = sum(w_i * y_i)
            Y_w = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update paths
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (y_w - mean)/sigma
            # (y_w - mean)/sigma = B @ z_w where z_w = sum w_i z_i
            # We need C^(-1/2) * (y_w - mean)/sigma
            # Since C = B B^T, C^(-1/2) = B^(-1)
            # So C^(-1/2) * B @ z_w = B^(-1) B z_w = z_w
            # This simplifies things greatly!
            # z_w = sum w_i z_i
            # Note: z_i are the standard normal samples used to generate y_i
            # We need to keep track of Z_sorted corresponding to Y_sorted
            Z_sorted = Z[idx]
            Z_w = np.sum(weights[:, np.newaxis] * Z_sorted[:mu], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Z_w
            
            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp(cs / (n + 1) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # ph = (1-c1)*ph + sqrt(c1*(2-c1)*mueff) * C^(-1/2) * (y_w - mean)/sigma
            # C^(-1/2) * (y_w - mean)/sigma = Z_w
            ph = (1 - c1) * ph + np.sqrt(c1 * (2 - c1) * mueff) * Z_w
            
            # hsig
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cc)**(2 * (eval_count // lam))) < (1 + 2.0 / (n + 1)) * np.sqrt((n + 1) * (n - 1) + 1))
            
            # Delta term
            delta = (1 - hsig) * c1 * (2 - c1) * np.outer(ph, ph)
            
            # Rank mu update
            # C = (1 - c1 - cmu) * C + c1 * (ps ps^T + (1-hsig)*c1(2-c1)*ph ph^T) + cmu * sum w_i (y_i - mean)(y_i - mean)^T / sigma^2
            # The term sum w_i (y_i - mean)(y_i - mean)^T / sigma^2
            # (y_i - mean)/sigma = B @ z_i
            # So (y_i - mean)(y_i - mean)^T / sigma^2 = B @ z_i z_i^T @ B^T
            # We need sum w_i B @ z_i z_i^T @ B^T = B @ (sum w_i z_i z_i^T) @ B^T
            
            Z_w_outer = np.zeros((n, n))
            for i in range(mu):
                Z_w_outer += weights[i] * np.outer(Z_sorted[i], Z_sorted[i])
            
            rank_mu = cmu * (B @ Z_w_outer @ B.T)
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + delta) + rank_mu
            
            # Update B and D
            # Cholesky decomposition
            try:
                B_new = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-10 * np.eye(n)
                B_new = np.linalg.cholesky(C)
                
            # Update D (sqrt det C)
            # det(C) = prod(diag(B_new)^2)
            # D = sqrt(det(C)) = prod(diag(B_new))
            D = np.prod(np.diag(B_new))
            
            # Update B for next iteration?
            # In CMA-ES, we usually keep B and D separate to avoid repeated Cholesky?
            # Actually, the standard implementation updates C, then computes B and D from C.
            # But we use B to generate Y. So we must update B.
            B = B_new
            
            # Mean update
            mean = Y_w
            
        return best_x, best_f
