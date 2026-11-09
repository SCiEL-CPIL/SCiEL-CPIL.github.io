import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sampling
        # Use 10 points to find a good starting point for CMA-ES
        n_init = 10
        # Simple random or Halton? Random is fine for 10 points in 5D.
        init_points = rng.uniform(-5, 5, size=(n_init, dim))
        
        init_results = []
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(p)
            init_results.append((p, f))
            
        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])
        
        # Phase 2: CMA-ES Refinement
        # We will run CMA-ES on the best initial point.
        # If it stagnates, we try the next best.
        
        def run_cma_es(x0, f0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            # CMA-ES parameters
            lambda_ = 9
            mu = lambda_ // 2  # 4
            # Weights for parents
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
            
            h_sigma = (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff - 1) / (n + 1) + 1)
            
            m = x0.copy()
            sigma = 0.5  # Initial step size
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            calls_local = 0
            last_improvement = 0
            best_f_local = f0
            
            for gen in range(max_calls // lambda_ + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Generate population
                pop = []
                for i in range(lambda_):
                    # Cholesky decomposition of C
                    L = np.linalg.cholesky(C)
                    z = rng.standard_normal(n)
                    y = sigma * (L @ z)
                    x = m + y
                    x = np.clip(x, -5.0, 5.0)
                    f = evaluate(x)
                    pop.append((x, f))
                    calls_local += 1
                    
                # Sort by fitness
                pop.sort(key=lambda x: x[1])
                
                # Update best
                if pop[0][1] < best_f_local:
                    best_f_local = pop[0][1]
                    last_improvement = gen
                
                # Check stagnation
                if gen - last_improvement > 10:
                    break
                
                # Compute new mean
                m_new = np.zeros(n)
                for i in range(mu):
                    m_new += w[i] * pop[i][0]
                
                # Update path ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (m_new - m))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Update path pc
                pc = (1 - c_1) * pc + np.sqrt(c_1 * (2 - c_1) * mu_eff) / sigma * (m_new - m)
                
                # Update C
                h_c = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_1)**(2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff - 1) / (n + 1) + 1)
                
                C = (1 + c_1 * h_c * (1 / c_1 - 1 + c_mu) - c_1) * C
                # This formula is a bit complex, let's use the standard update:
                # C = (1 - c_1 - c_mu + c_1*h_c*(1/c_1 - 1 + c_mu)) * C + c_1 * pc @ pc.T + c_mu * sum(w_i * y_i @ y_i.T)
                
                # Actually, simpler standard update:
                C = (1 - c_1 - c_mu) * C
                if h_c:
                    C += c_1 * (pc @ pc.T + (1/c_1 - 1 + c_mu) * np.eye(n))
                else:
                    C += c_1 * (pc @ pc.T)
                
                for i in range(mu):
                    y_i = (pop[i][0] - m) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                m = m_new
                
            return best_f_local

        # Try CMA-ES on top candidates
        remaining = budget - calls_made
        # Allocate 80% of remaining to the best candidate
        # If it fails, use the rest for the second best
        budgets = [int(remaining * 0.8), int(remaining * 0.15), int(remaining * 0.05)]
        
        for i in range(min(3, len(init_results))):
            if time.time() > deadline or calls_made >= budget:
                break
            x0, f0 = init_results[i]
            # Only run if we haven't found a perfect solution (f_opt is unknown, but if f0 is very low, maybe stop?)
            # Just run CMA-ES
            run_cma_es(x0, f0, budgets[i] if i < len(budgets) else 10)

        return best_x, best_f
