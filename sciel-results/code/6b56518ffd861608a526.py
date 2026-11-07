import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        # Phase 1: Initial Exploration
        n_init = 50
        if self.budget <= n_init:
            n_init = self.budget
            
        X_init = np.random.uniform(-5, 5, (n_init, self.dim))
        F_init = np.zeros(n_init)
        
        for i in range(n_init):
            F_init[i] = func(X_init[i])
            
        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx]
        f_best = F_init[best_idx]
        
        evals_used = n_init
        
        # Phase 2: CMA-ES
        remaining_budget = self.budget - evals_used
        
        if remaining_budget > 10:
            x_best, f_best = self._run_cma_es(func, x_best, f_best, remaining_budget, deadline)
            
        return x_best, f_best

    def _run_cma_es(self, func, x0, f0, budget, deadline):
        n = self.dim
        N = 4 + int(3 * np.log(n))
        mu = N // 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) * (n + 1) / mu_eff) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 3 * mu_eff / n)
        c_cm = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(c_1, 2 * (1 - c_1) * (mu_eff - 2 + np.sqrt((mu_eff - 2)**2 - mu_eff * (n + 2 - mu_eff))) / (mu_eff + (n + 2)**2))
        
        m = x0.copy()
        sigma = 1.0
        C = np.eye(n)
        B, D = np.linalg.eigh(C)
        D_inv = 1.0 / D
        p_c = np.zeros(n)
        p_s = np.zeros(n)
        
        evals = 0
        best_x = m
        best_f = f0
        
        while evals + N <= budget:
            if time.time() > deadline - 5:
                break
                
            Y = np.zeros((N, n))
            for i in range(N):
                z = np.random.randn(n)
                Y[i] = m + sigma * (B @ (D * z))
            
            F = np.zeros(N)
            for i in range(N):
                F[i] = func(Y[i])
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
            evals += N
            
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            
            # Calculate weighted sum of selected steps (in standard normal space)
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * (Y_sorted[i] - m) / sigma
                
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update p_s
            p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D_inv * (B.T @ y_w)))
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) / (c_c * (1 - c_c))
            for i in range(mu):
                y_i = (Y_sorted[i] - m) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i) / (c_sigma * (1 - c_sigma))
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals + 1))) - 1))
            
            # Update mean
            m = m + c_cm * sigma * y_w
            
            # Update B, D
            B, D = np.linalg.eigh(C)
            D_inv = 1.0 / D
            
            if sigma < 1e-12:
                break
                
        return best_x, best_f
