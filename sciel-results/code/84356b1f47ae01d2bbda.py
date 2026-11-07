import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        xmean = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        
        calls = 0
        deadline = time.time() + 290
        f_best = np.inf
        x_best = None
        
        # Initial point
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls += 1
        xmean = x0.copy()
        f_best = f0
        x_best = x0.copy()
        
        gen = 0
        while calls < budget and time.time() < deadline:
            gen += 1
            # Generate and evaluate population
            Y = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = xmean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                Y.append((fy, y))
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            
            if len(Y) < 2:
                break
            
            Y.sort(key=lambda x: x[0])
            Y_sel = np.array([y[1] for y in Y[:mu]])
            
            # Update mean
            xmean_old = xmean.copy()
            xmean = np.sum(w[:, None] * Y_sel, axis=0)
            
            # Update sigma
            y_sigma = (xmean - xmean_old) / (sigma * np.sqrt(mueff))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            hsig = min(1, np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*gen)) / (1.4 + 1/dim) / chiN)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1) * hsig)
            
            # Update C
            yc = (xmean - xmean_old) / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * yc
            delta_h = (1 - hsig) * cc * (2 - cc) * np.dot(pc, pc) / dim
            C = (1 + c1 * (cmu * np.outer(yc, yc) - cc) + c1 * delta_h) * C + cmu * np.sum(w[:, None, None] * np.outer(Y_sel - xmean_old, Y_sel - xmean_old) / sigma**2, axis=0)
            
            # Decompose C
            D2, B = np.linalg.eigh(C)
            D2 = np.clip(D2, 1e-20, None)
            D = np.sqrt(D2)
        
        return x_best, f_best
