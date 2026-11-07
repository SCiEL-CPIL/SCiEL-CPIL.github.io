import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        import time
        start_time = time.time()
        deadline = start_time + 290  # 10s margin

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 4 * (3 + int(np.log2(dim)))  # 20 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # State
        x0 = rng.uniform(-5, 5, dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        
        # Cholesky factor of C
        B = np.eye(dim)
        D = np.ones(dim)
        
        best_x = x0
        best_f = func(x0)
        calls = 1
        
        while calls + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # x = x0 + sigma * B @ (D * z)
                x = x0 + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                pop.append((fx, x))
                
            # Sort by fitness
            pop.sort(key=lambda p: p[0])
            
            # Update best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1]
                
            # Selection
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop[i][1]
                
            # Update mean
            x0 = x_new
            
            # Update step size
            y = (x_new - x0) / sigma  # Wait, x0 was updated. Need old x0.
            # Correction:
            # Let m_old = x0_before_update
            # I need to store m_old before updating x0.
            
            # Let's restructure the loop to store m_old.
            pass
