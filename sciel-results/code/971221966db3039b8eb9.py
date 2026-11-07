import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Evaluator kills at 330s. We use 300s as a hard safety limit to ensure we return.
        # The task says "keep an EXPLICIT deadline guard".
        deadline = start_time + 300
        
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- CMA-ES Implementation ---
        # Parameters for CMA-ES
        # For dim=5, lambda = 4 + floor(3*log(dim)) = 4 + 6 = 10 is standard.
        # Let's use lambda=8 to allow more generations within 500 budget (500/8 = 62 generations).
        # Or lambda=10 -> 50 generations. 
        # Given the extreme anisotropy, a slightly larger population might help capture the direction 
        # but smaller allows faster convergence once direction is found.
        # Let's stick with lambda=8 for a balance.
        
        lambda_ = 8
        
        # Initial state
        x = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Update rates
        mu = lambda_ // 2  # Number of weighted individuals
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step sizes for mean and covariance
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff/dim - 1)) + c_sigma
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        d_c = 1 + c_c + 2 * max(0, sqrt((mu_eff-1)/(dim+1) - 1))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factor update state
        # We will use a simple rank-one and rank-mu update for C
        # and maintain a Cholesky factor L such that C = L @ L.T
        
        # Initialize L
        L = np.eye(dim)
        
        # Initial population evaluation is handled in the loop
        
        while can_call():
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = x + sigma * (L @ z)
            
            # Evaluate population
            f_vals = np.zeros(lambda_)
            for i in range(lambda_):
                f_vals[i] = evaluate(pop[i])
                if not can_call():
                    break
            
            if not can_call():
                break
                
            # Sort by fitness
            order = np.argsort(f_vals)
            pop_sorted = pop[order]
            f_sorted = f_vals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Weighted mean of top mu
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * (pop_sorted[i] - x) / sigma
            
            # Update p_sigma
            h_sigma = 1.0 if (np.sqrt(1 - (1+c_sigma)**(-2*(calls_made+1)/mu_eff)) * 
                              np.linalg.norm(y_w) / np.linalg.norm(L)) < (1.4 + 2/(dim+1)) * np.sqrt(mu_eff + 1 - mu_eff) else 0.0
            # Simplified h_sigma check (standard CMA-ES)
            # Actually, let's use the standard heuristic for h_sigma
            # h_sigma = 1 if norm(y_w) < (1.4 + 2/(dim+1)) * sqrt((1 - (1+c_sigma)**(-2*(gen+1)/mu_eff)) * (mu_eff+1-mu_eff)) * norm(L)
            # Since we don't track gen exactly in a simple loop, we approximate.
            # For dim=5, this is usually stable. Let's use a simpler check or just set h_sigma=1 
            # if the step is not too large relative to sigma.
            # A robust simple implementation often just updates p_sigma always or uses a fixed threshold.
            # Let's use the standard formula but we need to be careful with L norm.
            
            # Standard CMA-ES h_sigma calculation:
            # h_sigma = 1.0 if np.linalg.norm(y_w) / np.linalg.norm(L) < (1.4 + 2.0/(dim+1.0)) * np.sqrt((1.0 - (1.0 + c_sigma)**(-2.0 * (calls_made/mu_eff))) * (mu_eff + 1.0 - mu_eff)) else 0.0
            # Note: (1 - (1+c)^(-2t)) approx 2t c for small t.
            # Let's just use h_sigma = 1.0 for simplicity in this specific constrained budget context, 
            # or better, implement the check properly.
            
            # Let's implement the check properly.
            # We need to track generation count for the (1+c_sigma)^(-2*gen/mu_eff) term.
            # Let's assume gen = calls_made // lambda_
            gen = calls_made // lambda_
            # The term (1 - (1+c_sigma)^(-2*gen/mu_eff))
            # If gen is small, this is small.
            
            # Actually, a simpler and very robust variant for small budgets is to skip the h_sigma check 
            # and just update p_sigma, or use a fixed h_sigma=1 if the step is reasonable.
            # However, to be safe and standard:
            
            norm_L = np.linalg.norm(L)
            # If L is identity initially, norm_L = sqrt(dim)
            
            # Let's use a simplified h_sigma that is 1 unless the step is huge.
            # If y_w is large, it means we moved a lot.
            # Standard: h_sigma = 1 if ||y_w|| < (1.4 + 2/(d+1)) * sqrt(1 - (1+c_s)^(-2*gen/m_eff)) * sqrt(m_eff + 1 - m_eff) * ||L||
            # Wait, sqrt(m_eff + 1 - m_eff) = 1.
            # So threshold = (1.4 + 2/(d+1)) * sqrt(1 - (1+c_s)^(-2*gen/m_eff)) * ||L||
            
            # If gen=0, sqrt(1-1)=0. So h_sigma=0 initially? No, usually it's 1 at start.
            # The formula is for the expected norm.
            # Let's just set h_sigma = 1.0 for the first few generations to allow adaptation.
            if gen < 5:
                h_sigma = 1.0
            else:
                # Calculate threshold
                exp_term = 1 - (1 + c_sigma)**(-2 * gen / mu_eff)
                if exp_term < 0: exp_term = 0
                threshold = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(exp_term) * norm_L
                h_sigma = 1.0 if np.linalg.norm(y_w) < threshold else 0.0

            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Update x
            x = x + sigma * y_w
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update C
            # Rank-one update
            C += c_c * (2 - c_c) * (p_c @ p_c.T)
            
            # Rank-mu update
            # C += c_c * (2 - c_c) * mu_eff * (1 - (1+c_c)**(-2*gen/mu_eff)) * sum(w_i * (y_i - y_w)(y_i - y_w)^T)
            # Let's simplify the rank-mu update weight.
            # w_mu = c_c * (2 - c_c) * mu_eff
            # The factor (1 - (1+c_c)**(-2*gen/mu_eff)) is close to 1 for large gen, small for small gen.
            # Let's use a fixed weight for rank-mu to keep it simple and stable.
            # C += c_c * (2 - c_c) * mu_eff * sum(...)
            
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pop_sorted[i] - x) / sigma  # This is y_i relative to OLD x? 
                # Wait, y_w was calculated relative to OLD x.
                # The formula uses (y_i - y_w) where y_i = (x_i - x_old)/sigma
                # So we need y_i for the top mu individuals.
                # We have pop_sorted. x_old is the x before update.
                # We updated x already. So we need to store x_old.
                
            # Let's restart the loop logic slightly to store x_old.
            # Actually, I can just calculate y_i now using the current x (which is new) and subtract y_w? No.
            # y_i = (x_i - x_old) / sigma.
            # y_w = sum w_i y_i.
            # I updated x = x_old + sigma * y_w.
            # So x_old = x - sigma * y_w.
            # y_i = (pop_sorted[i] - (x - sigma * y_w)) / sigma = (pop_sorted[i] - x)/sigma + y_w.
            
            for i in range(mu):
                y_i = (pop_sorted[i] - x) / sigma + y_w
                diff = y_i - y_w
                rank_mu_term += w[i] * np.outer(diff, diff)
            
            C += c_c * (2 - c_c) * mu_eff * rank_mu_term
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
            
            # Update L (Cholesky)
            # We need to update L such that L @ L.T = C
            # Simple approach: Recompute Cholesky every generation? 
            # For dim=5, this is very cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            # Clip sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Early stopping check
            if sigma < 1e-5:
                # If sigma is tiny, we are likely converged.
                # We can break or continue. Breaking saves time.
                # But maybe we can jump? No, let's just break.
                break

        return best_x
