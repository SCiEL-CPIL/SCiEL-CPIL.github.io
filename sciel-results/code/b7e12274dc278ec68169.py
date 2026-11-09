import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B, D = np.linalg.eigh(C)
        D = np.sqrt(D)
        pc = np.zeros(d)
        ps = np.zeros(d)
        hsigma = np.sqrt((1 - np.exp(-2 / (d + 1))) / (d + 1)) + 1
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample
            Y = np.random.randn(lam, d)
            # x_i = x + sigma * B * D * Y_i
            # Y_i is row vector.
            # Z = Y * D
            Z = Y * D
            # W = Z @ B.T
            W = Z @ B.T
            Pop = x + sigma * W
            Pop = np.clip(Pop, -5, 5)
            
            # 2. Evaluate
            Fs = np.array([func(p) for p in Pop])
            calls += lam
            
            # 3. Update Best
            idx_min = np.argmin(Fs)
            if Fs[idx_min] < best_f:
                best_f = Fs[idx_min]
                best_x = Pop[idx_min].copy()
                
            # 4. Sort
            SortIdx = np.argsort(Fs)
            X_sorted = Pop[SortIdx]
            
            # 5. Update Mean
            X_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 6. Update ps
            # ps = (1 - c_sigma) * ps + c_sigma * hsigma * (C^-1/2 * (X_new - X) / sigma)
            # C^-1/2 = D^-1 @ B.T
            Step = (X_new - x) / sigma
            Step_Unrotated = Step @ B.T
            Step_Uncorr = Step_Unrotated / D
            ps = (1 - c_sigma) * ps + c_sigma * hsigma * Step_Uncorr
            
            # 7. Update pc
            pc = (1 - c_c) * pc + c_c * (X_new - x)
            
            # 8. Update Sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(d) - hsigma))
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # 9. Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * Sum(w_i * (Y_i - Y_w) @ (Y_i - Y_w).T)
            # Note: The sum term uses the selected steps in the unrotated space?
            # Actually, the term is: sum w_i * (z_i - z_w) @ (z_i - z_w).T
            # where z_i = C^-1/2 * (x_i - x) / sigma
            # z_i = D^-1 @ B.T @ (x_i - x) / sigma
            
            Z_selected = []
            for i in range(mu):
                z = (X_sorted[i] - x) / sigma
                z_unrotated = z @ B.T
                z_uncorr = z_unrotated / D
                Z_selected.append(z_uncorr)
            Z_selected = np.array(Z_selected)
            
            Z_w = np.sum(weights[:, np.newaxis] * Z_selected, axis=0)
            Diff = Z_selected - Z_w
            OuterSum = np.sum(weights[:, np.newaxis, np.newaxis] * Diff[:, np.newaxis, :] * Diff[:, :, np.newaxis], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C) + c_mu * (B @ OuterSum @ B.T)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # 10. Eigen-decompose
            D_eig, B = np.linalg.eigh(C)
            # Clip negative eigenvalues
            D_eig = np.clip(D_eig, 1e-10, None)
            D = np.sqrt(D_eig)
            # B is already orthogonal from eigh
            
            # 11. Update X
            x = X_new
            
        return best_x, best_f
