import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        calls = 0
        best_f = np.inf
        best_x = None

        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        if lam % 2 != 0:
            lam += 1
        if lam > budget:
            lam = max(2, budget)
            
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (n + 2) - 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        x_mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        
        # Initial decomposition
        vals, vecs = np.linalg.eigh(C)
        D = np.sqrt(vals)
        B = vecs
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        w_mu = w - 1.0 / mu_eff

        gen = 0
        while calls + lam <= budget:
            ys = []
            xs = []
            for _ in range(lam):
                z = np.random.normal(0, 1, n)
                y = B @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                ys.append(y)
                xs.append(x)
            
            fs = np.array([wrapped_func(x) for x in xs])
            
            idx = np.argsort(fs)
            xs_sorted = [xs[i] for i in idx]
            ys_sorted = [ys[i] for i in idx]
            
            x_new_mean = np.sum([w[i] * xs_sorted[i] for i in range(mu)], axis=0)
            
            # Step size update
            y_sigma = (x_new_mean - x_mean) / (sigma * np.linalg.norm(D)) if np.linalg.norm(D) > 0 else np.zeros(n)
            # Note: The standard CMA-ES update uses D in the denominator for y_sigma calculation 
            # usually defined as y_sigma = (x_new_mean - x_mean) / sigma. 
            # But the path update for p_sigma uses y_sigma.
            # Let's stick to the standard formulation:
            # y_sigma = (x_new_mean - x_mean) / sigma
            
            y_sigma = (x_new_mean - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma)**2) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Covariance update
            C = (1 + c_1 * (2 - c_1) * h_sigma - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum([w_mu[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu)], axis=0)
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) - 1))
            
            x_mean = x_new_mean
            
            # Decompose
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 1e-12, None)
            D = np.sqrt(vals)
            B = vecs
            
            gen += 1
            
            if sigma < 1e-12:
                break

        # If we have remaining calls, use them for a simple local search
        if best_x is not None and calls < budget:
            # Simple hill climb
            step = 0.1
            while calls < budget:
                improved = False
                for i in range(n):
                    if calls >= budget: break
                    for sign in [1, -1]:
                        if calls >= budget: break
                        x_try = best_x.copy()
                        x_try[i] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = wrapped_func(x_try)
                        if f_try < best_f:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
