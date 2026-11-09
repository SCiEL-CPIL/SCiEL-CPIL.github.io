import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Initialize at origin
        x = np.zeros(d)
        try:
            f_x = safe_func(x)
        except StopIteration:
            return best_x, best_f

        # CMA-ES Parameters
        # Using lambda=4 as per directive, but might be too small for d=5.
        # Standard is lambda = 4 + 3*ln(d). For d=5, lambda=11.
        # Let's try lambda=6 as a compromise or stick to 4?
        # The directive says "Strategy to try... population size of 4".
        # I will use lambda=4 but implement a robust restart.
        
        lam = 4
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size adaptation
        c_sigma = (lam / 2 + 1) / (d + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + lam / d) / (d + 4 + 2 * lam / d)
        c_1 = 2 / ((d + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (lam / 2 - 2 + 1 / lam) / ((d + 2) ** 2 + lam))
        
        # Initial state
        mean = np.zeros(d)
        sigma = 2.0 # Start with a reasonable step size for [-5,5] box
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Precompute Cholesky or use SVD for C_sqrt
        def update_chol(C):
            # Ensure positive definite
            L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
            return L.T # We want C = L.T @ L, so C_sqrt = L.T? 
            # Actually, we need z ~ N(0, C). If C = A A^T, then x = mean + sigma * A @ z.
            # np.linalg.cholesky returns L such that C = L L^T.
            # So A = L.
            return L

        C_sqrt = update_chol(C)
        h_sig = (c_sigma + 1) / (c_sigma + d + 1) # Approximation for h_sigma threshold
        
        # Main Loop
        max_generations = int(budget / lam)
        stall_count = 0
        stall_limit = 20
        
        for gen in range(max_generations):
            if time.time() > deadline or calls + lam > budget:
                break
                
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.normal(0, 1, d)
                y = C_sqrt @ z
                x_i = mean + sigma * y
                pop.append(x_i)
            
            # Evaluate
            f_vals = []
            for x_i in pop:
                try:
                    f_i = safe_func(x_i)
                    f_vals.append(f_i)
                except StopIteration:
                    break
            
            if len(f_vals) < lam:
                break
                
            # Sort
            idx = np.argsort(f_vals)
            pop = [pop[i] for i in idx]
            f_vals = [f_vals[i] for i in idx]
            
            # Update Mean
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * np.array(pop), axis=0)
            
            # Update p_sigma
            y_mean = (mean - mean_old) / (sigma * np.linalg.norm(C_sqrt @ np.ones(d))) # Normalization factor?
            # Standard formula: y_k = (mean - mean_old) / sigma
            # But we need to normalize by sqrt of trace? No, just sigma.
            y_k = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_k
            
            # Check h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_lam = np.sqrt(d) * (1 - 1 / (2 * d * d) + 1 / (8 * d**4))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * chi_lam else 0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_k
            
            # Update Covariance Matrix
            # Delta H = c_1 * (p_c p_c^T + (1 - h_sigma) * (1 + (1 - c_c) * (1 - c_c) - mu_eff) * C - C)
            # + c_mu * sum(w_i * (y_i - y_w)(y_i - y_w)^T)
            
            # y_i in C-space
            # y_i = (x_i - mean_old) / sigma
            # y_w = mean of weighted y_i? No, y_w is the weighted mean of the selected y_i?
            # Actually, the formula uses the selected points.
            
            # Let's compute the rank-mu update
            # y_sel = [ (pop[i] - mean_old) / sigma for i in range(lam) ]
            # But we need them in the C-space? No, the formula is for the covariance of the search distribution.
            # The update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * (1 + (1 - c_c)*(1 - c_c) - mu_eff) * C - C) 
            # Wait, the standard formula is:
            # C <- (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) (1 + (1 - c_c)(1 - c_c) - mu_eff) C) 
            # + c_mu sum_{i=1}^{lam} w_i (y_i - y_w)(y_i - y_w)^T
            
            # Where y_i are the selected steps in the C-space?
            # Actually, y_i = (x_i - mean_old) / sigma.
            # And y_w = sum w_i y_i.
            
            y_sel = np.array([(pop[i] - mean_old) / sigma for i in range(lam)])
            y_w = np.sum(w[:, np.newaxis] * y_sel, axis=0)
            
            # Rank-mu update term
            rank_mu = np.zeros((d, d))
            for i in range(lam):
                diff = y_sel[i] - y_w
                rank_mu += w[i] * np.outer(diff, diff)
                
            # Rank-one update term
            rank_one = np.outer(p_c, p_c)
            # The term (1 - h_sigma) * (1 + (1 - c_c)*(1 - c_c) - mu_eff) * C
            # This is often simplified or handled carefully to maintain PD.
            # A simpler, more stable update often used in implementations:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * (1 + (1 - c_c)*(1 - c_c) - mu_eff) * C) + c_mu * rank_mu
            # This can lead to non-PD matrices if not careful.
            # Let's use the "Damped" update or ensure symmetry and PD.
            
            # Simpler approach for small d:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - h_sigma) * (1 + (1 - c_c) * (1 - c_c) - mu_eff) * C) + c_mu * rank_mu
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 + (1 - c_c) * (1 - c_c) - mu_eff) * C) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure Positive Definite
            try:
                C_sqrt = update_chol(C)
            except np.linalg.LinAlgError:
                # If not PD, restart
                break
            
            # Update Step Size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lam - 1))
            
            # Check for stall
            if f_vals[0] < best_f:
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > stall_limit:
                # Restart from best with larger sigma
                mean = best_x.copy()
                sigma *= 2.0
                C = np.eye(d)
                C_sqrt = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                stall_count = 0

        # Final Polish with (1+1)-ES
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
