import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # For n=5, standard lambda is 4 + floor(3*log(n)) = 4+4 = 8. 
        # We use lambda=10 for slightly more diversity in early stages.
        lam = 10
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        # CMA-ES standard: sigma = 1.0 initially for box [-5,5] might be too small?
        # The box is [-5,5]. Center is 0. 
        # If we start at 0, sigma=1.0 covers [-1,1] roughly. 
        # Better to start with a larger sigma to explore the whole box quickly.
        # However, CMA-ES adapts sigma. 
        # Let's start with sigma = 2.0 to cover a good portion of the box [-5,5] (2.5 sigma ~ 5).
        sigma = 2.0
        
        # Mean
        m = np.zeros(n)
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Precompute inverse sqrt of C for efficiency (optional, but good for correctness)
        # Actually, we need to sample from N(m, sigma^2 C).
        # We can use Cholesky decomposition each generation or update it.
        # For n=5, Cholesky is cheap.
        
        # Step size control parameters
        # Standard CMA-ES uses cs, ds, cc, cm.
        # For simplicity and robustness in low dim, we can use a simplified version 
        # or standard parameters.
        # Let's use standard CMA-ES parameters from Hansen's paper.
        
        # Evolution path parameters
        # ps = 0
        # pc = 0
        # damps = 1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + cs
        # cs = (mu_eff+2)/(n+mu_eff+5)
        # damps = 1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + cs
        # cm = 1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + 2*cs*(damps-1)
        # cc = (4-mu_eff)/(n+4)
        
        # Let's compute them
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        cm = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + 2 * cs * (damps - 1)
        cc = (4 - mu_eff) / (n + 4)
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # History for AOCC
        # We don't need to store history, just return best_x, best_f
        # But the task says "return value of __call__ is ignored".
        # Wait, "Every call is recorded... return value of __call__ is ignored."
        # So we just need to make the calls.
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x[:n]
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val
        
        # Initial evaluation of the mean? 
        # CMA-ES usually starts by sampling from N(m, sigma^2 I).
        # We don't need to evaluate m itself necessarily, but it helps if it's good.
        # Let's just start the loop.
        
        while calls_made + lam <= budget and time.time() < deadline:
            # Sample lambda points
            # Need to compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset or add jitter
                C = np.eye(n) + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            ys = []
            xs = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = L @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                ys.append(y)
                xs.append(x)
                
            # Evaluate
            fs = np.array([safe_func(x) for x in xs])
            
            # Sort indices by function value
            order = np.argsort(fs)
            ys = np.array(ys)[order]
            xs = np.array(xs)[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Update mean
            m_new = m + sum(weights[i] * (xs[i] - m) for i in range(mu))
            
            # Update covariance
            # C = (1 - cm*cm) * C + cm * ( (1 + (1-cs)*(1+cs)) * sum(w_i (y_i - m_y)(y_i - m_y)^T) + (1-cs)^2 * ps ps^T )
            # Wait, the standard update is:
            # C = (1 - cm*cm) C + cm ( (1 + (1-cs)(1+cs)) H + (1-cs)^2 ps ps^T )
            # Where H = sum_{i=1}^mu w_i (y_i - y_bar)(y_i - y_bar)^T
            # And y_bar = sum_{i=1}^mu w_i y_i
            # And ps = (1-cs) ps + sqrt(cs(2-cs)mu_eff) (C^{-1/2})(m_new - m)/sigma
            
            # Compute y_bar
            y_bar = np.sum(ys[:mu] * weights[:, np.newaxis], axis=0)
            
            # Compute H
            H = np.zeros((n, n))
            for i in range(mu):
                dy = ys[i] - y_bar
                H += weights[i] * np.outer(dy, dy)
                
            # Update ps
            # C^{-1/2} (m_new - m) / sigma
            # We have L such that C = L L^T. So C^{-1/2} = L^{-T}
            # Solve L^T v = (m_new - m) / sigma
            diff = (m_new - m) / sigma
            try:
                v = np.linalg.solve(L.T, diff)
            except np.linalg.LinAlgError:
                v = diff # Fallback
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * v
            
            # Check for sigma update based on ps
            # sigma = sigma * exp(cs/damps * (||ps||/chiN - 1))
            chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / damps * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (m_new - m) / sigma
            
            # Update C
            # Term 1: (1 - cm*cm) * C
            # Term 2: cm * ( (1 + (1-cs)*(1+cs)) * H + (1-cs)^2 * outer(ps, ps) )
            factor_H = 1 + (1 - cs) * (1 + cs)
            term2 = factor_H * H + (1 - cs)**2 * np.outer(ps, ps)
            C = (1 - cm * cm) * C + cm * term2
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update mean
            m = m_new
            
            # Optional: Check convergence
            # If sigma is very small or range of fs is very small, we might stop
            # But we have budget, so keep going unless time is up.
            if sigma < 1e-6:
                # If sigma is tiny, we might be stuck. 
                # We could restart, but for simplicity, just break or continue with tiny steps.
                # Given the "sharp ridge", getting stuck in a flat area is possible.
                # However, CMA-ES usually handles this by increasing sigma if ps is large.
                # If ps is small, sigma decreases.
                # If we are at a minimum, sigma should decrease.
                pass

        return best_x, best_f
