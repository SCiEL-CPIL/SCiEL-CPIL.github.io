import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(20*n*n))
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        for gen in range(budget - 1):
            # Generate population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if needed
                C += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            Y = []
            for i in range(lam):
                z = np.random.standard_normal(n)
                y = x + sigma * (L @ z)
                Y.append(y)
            
            # Evaluate
            F = []
            for y in Y:
                fy = func(y)
                F.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Sort
            order = np.argsort(F)
            Y_sorted = [Y[i] for i in order]
            F_sorted = [F[i] for i in order]
            
            # Update mean
            y_w = np.sum(weights[:, np.newaxis] * np.array(Y_sorted[:mu]), axis=0)
            x_old = x.copy()
            x = x + y_w
            
            # Update C
            # y_inv_cov: steps in normalized coordinates
            # y_i = (Y_sorted[i] - x_old) / sigma
            # But we need the weighted sum of outer products.
            # H = sum w_i y_i y_i^T
            # C = (1-c1-cmu)C + c1(pc pc^T + (1-delta_sig) I) + cmu H
            
            # Calculate H
            H = np.zeros((n, n))
            for i in range(mu):
                y_step = (Y_sorted[i] - x_old) / sigma
                H += weights[i] * np.outer(y_step, y_step)
                
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (y_w / sigma)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (y_w / sigma)
            
            # Delta sigma
            delta_sig = 1 + 2 * max((mu_eff - 1) / (n + mu_eff - 1), 0) * (np.linalg.norm(ps) / chi_n - 1)
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_sig - 1) * C) + cmu * H
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Termination
            if sigma < 1e-12:
                break
                
        return best_x, best_f
