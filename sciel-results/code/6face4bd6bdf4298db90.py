import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        lam = 10
        mu = lam // 2
        
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        m = np.zeros(n)
        sigma = 2.5
        C = np.eye(n)
        Cinv = np.eye(n)
        
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        calls_made = 0
        best_x = m
        best_f = np.inf
        
        while calls_made + lam <= budget:
            ys = np.random.multivariate_normal(np.zeros(n), C, size=lam)
            xs = m + sigma * ys
            xs = np.clip(xs, -5, 5)
            
            fs = np.array([func(x) for x in xs])
            calls_made += lam
            
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            m_new = np.sum(w[:, None] * xs[:mu], axis=0)
            
            yk = (xs[:mu] - m) / sigma
            C_update = np.sum(w[:, None, None] * yk[:, :, None] * yk[:, None, :], axis=0)
            C_new = C + c1 * (np.outer(pc, pc) - C) + cmu * C_update
            
            try:
                Cinv_new = np.linalg.inv(C_new)
            except np.linalg.LinAlgError:
                C_new = C
                Cinv_new = Cinv
            
            ps_new = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_new @ (m_new - m) / sigma)
            
            hsig = np.linalg.norm(ps_new) / np.sqrt(1 - (1 - cs)**(2 * (calls_made // lam + 1))) < (1.4 + 2 / (n + 1)) * cs
            pc_new = (1 - cc) * pc + cc * hsig * ps_new
            
            sigma_new = sigma * np.exp(cs / ds * (np.linalg.norm(ps_new) / np.sqrt(2 * np.log(2)) - 1))
            sigma_new = np.clip(sigma_new, 1e-8, 10.0)
            
            m = m_new
            C = C_new
            Cinv = Cinv_new
            sigma = sigma_new
            ps = ps_new
            pc = pc_new

        while calls_made < budget:
            y = np.random.multivariate_normal(np.zeros(n), C, size=1)[0]
            x_new = np.clip(m + sigma * y, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return best_x, best_f
