        # 2. CMA-ES Refinement
        remaining_budget = budget - count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
            
        # CMA-ES Parameters
        lam = 4 + 3 * int(np.log(n))
        if lam < 4: lam = 4
        if lam > 20: lam = 20
        
        sigma = 1.0 
        m = best_x.copy()
        C = np.eye(n)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (n + 4.3)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(n)
        
        gen = 0
        while count < budget and time.time() < deadline:
            ys = []
            fs = []
            for k in range(lam):
                if count >= budget or time.time() > deadline:
                    break
                z = np.random.normal(0, 1, n)
                y = B @ z
                x = m + sigma * y
                f = func(x)
                count += 1
                ys.append(y)
                fs.append(f)
            
            if not fs:
                break
                
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = (m + sigma * ys[0]).copy()
            
            new_m = m + sigma * sum(weights[k] * ys[k] for k in range(mu))
            
            B_inv = np.linalg.inv(B)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ (new_m - m)) / sigma
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
            threshold = (1.4 + 2 / (n + 1)) * denom
            h_sig = 1.0 if norm_p_sigma < threshold else 0.0
            
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_m - m) / sigma
            
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * sum(weights[k] * np.outer(ys[k], ys[k]) for k in range(mu))
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(n)
            
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
            
            m = new_m
            gen += 1
            
        return best_x, best_f
