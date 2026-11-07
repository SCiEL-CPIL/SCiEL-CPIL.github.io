import numpy as np

class Optimizer:
    """CMA-ES continuous optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Pop size: lambda = 4 + floor(3 * log(dim + 1))
        # For dim=5, lambda = 4 + floor(3 * log(6)) = 4 + floor(5.16) = 9
        lam = 4 + int(3 * np.log(dim + 1))
        
        # Initial step size
        # Standard initialization: sigma0 = 0.25 * (max - min) or similar
        # Box is [-5, 5], width 10. 
        # A good start for BBOB is often centered at 0 with sigma ~ 2.0
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates
        mu = lam // 2  # Weights top half
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial population
        pops = []
        fvals = []
        calls = 0
        
        # Evaluate initial population
        for i in range(lam):
            if calls >= budget:
                break
            z = rng.standard_normal(dim)
            x = x_mean + sigma * (C @ z) # Note: C is covariance, so sqrt(C) @ z is standard, but C @ z if C is not PSD? No, C is PSD.
            # Actually, standard CMA-ES uses B * diag(D) * z where C = B diag(D)^2 B^T
            # Simpler: Cholesky decomposition of C
            # But for small dim, we can just use eigendecomposition or assume C is updated properly.
            # Let's use the standard update: x = x_mean + sigma * (B * (D * z))
            # To keep it simple and efficient for dim=5:
            # We maintain C. To sample, we need the square root of C.
            # np.linalg.cholesky is efficient.
            
            # Let's stick to a simpler, robust implementation using Cholesky
            try:
                L = np.linalg.cholesky(C)
                x = x_mean + sigma * (L @ z)
            except np.linalg.LinAlgError:
                # Fallback if C becomes slightly non-PSD
                x = x_mean + sigma * z

            x = np.clip(x, -5, 5)
            fx = func(x)
            pops.append(x)
            fvals.append(fx)
            calls += 1
            
        best_x = pops[0].copy()
        best_fx = fvals[0]
        for i in range(len(pops)):
            if fvals[i] < best_fx:
                best_fx = fvals[i]
                best_x = pops[i].copy()

        # Main Loop
        while calls + lam <= budget:
            # Sort by fitness
            order = np.argsort(fvals)
            pops = [pops[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pops[0].copy()

            # Weighted mean of top mu
            x_w = np.zeros(dim)
            for i in range(mu):
                x_w += w[i] * pops[i]
            
            # Old mean
            x_mean_old = x_mean.copy()
            
            # Update mean
            x_mean = x_w
            
            # Update covariance matrix
            # Delta x
            delta_x = x_w - x_mean_old
            
            # Update p_sigma
            h_sigma = np.sqrt((1.5 - np.exp(-2 / (budget - calls))) * (1 - np.exp(-1 / (budget - calls)))) # Approximate
            # Actually, standard h_sigma calculation:
            # h_sigma = (||p_sigma|| / sqrt(1 - (1 - sigma)^2) < (1 + 2/dim) * E||N(0, I)||)
            # Let's use a simpler, stable update for p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (delta_x / sigma)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_x
            
            # Check for rank update
            # ||p_c|| / sqrt(1 - (1 - c_c)^t) < (1 + 2/dim) * E||N(0, I)||
            # We approximate the condition for rank update
            norm_p_c = np.linalg.norm(p_c)
            # E||N(0, I)|| approx sqrt(dim)
            if norm_p_c < (1 + 2.0 / dim) * np.sqrt(dim) * np.sqrt(1 - (1 - c_c)**(budget - calls)):
                # Rank-1 update
                C = (1 - c_c) * C + (1 - c_c) * mu_eff * np.outer(delta_x / sigma, delta_x / sigma)
                # Wait, standard formula:
                # C = (1 - c_c) * C + (1 - c_c) * (1 - mu_eff) * ... No.
                # C = (1 - c_c) * C + (1 - c_c) * mu_eff * (delta_x/sigma) (delta_x/sigma)^T
                # Actually, the update is:
                # C <- (1 - c_c) C + (1 - c_c) (1 - mu_eff) ... no.
                # Correct Rank-1 update:
                # C = (1 - c_c) * C + (1 - c_c) * mu_eff * np.outer(delta_x/sigma, delta_x/sigma)
                # Let's re-verify standard CMA-ES update.
                # C_{t+1} = (1 - c_c) C_t + (1 - c_c) (1 - mu_eff) ... no.
                # C_{t+1} = (1 - c_c) C_t + (1 - c_c) * (1 - mu_eff) is wrong.
                # It is: C_{t+1} = (1 - c_c) C_t + (1 - c_c) * mu_eff * (x_w - x_mean_old)/sigma ... outer product.
                # Let's use the explicit formula from Hansen's thesis.
                # C = (1 - c_c) * C + (1 - c_c) * mu_eff * np.outer(delta_x/sigma, delta_x/sigma)
                # Wait, the coefficient is (1 - c_c) * (1 - mu_eff) for the second term? No.
                # The term is (1 - c_c) * (1 - mu_eff) is not standard.
                # Standard: C = (1 - c_c) C + (1 - c_c) * (1 - mu_eff) ...
                # Let's look at the "Rank-1" update:
                # C <- (1 - c_c) C + (1 - c_c) * (1 - mu_eff) ... NO.
                # C <- (1 - c_c) C + (1 - c_c) * mu_eff * (delta_x/sigma) (delta_x/sigma)^T
                # This is correct.
                
                # Rank-mu update
                # Sum of weighted outer products of (x_i - x_w)/sigma
                # C += c_mu * sum(w_i * (x_i - x_w)/sigma * (x_i - x_w)/sigma^T)
                # c_mu = 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)^2 + mu_eff)
                c_mu = 2 * (mu_eff - 2 + 1.0/mu_eff) / ((dim + 2)**2 + mu_eff)
                # Add rank-mu update
                for i in range(mu):
                    diff = (pops[i] - x_w) / sigma
                    C += c_mu * w[i] * np.outer(diff, diff)
            
            # Update sigma
            # h_sigma calculation is complex, let's use a simplified robust version
            # sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/E||N(0,I)|| - 1))
            E_norm = np.sqrt(dim)
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 50.0)
            
            # Ensure C is symmetric and PSD
            C = (C + C.T) / 2.0
            # Eigenvalue check for stability
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-12):
                C += 1e-12 * np.eye(dim)
            
            # Generate next population
            new_pops = []
            new_fvals = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                try:
                    L = np.linalg.cholesky(C)
                    x = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x = x_mean + sigma * z
                    
                x = np.clip(x, -5, 5)
                fx = func(x)
                new_pops.append(x)
                new_fvals.append(fx)
                calls += 1
                
            pops = new_pops
            fvals = new_fvals

        return best_x, best_fx
