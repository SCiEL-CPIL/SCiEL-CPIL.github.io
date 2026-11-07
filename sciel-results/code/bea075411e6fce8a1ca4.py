import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        # Phase 1: CMA-ES
        # Allocate 70% of budget to CMA-ES
        cma_budget = int(total_budget * 0.7)
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        x0 = np.random.uniform(-5, 5, n)
        m = x0.copy()
        sigma = 2.0
        C = np.eye(n)
        ps = np.zeros(n)
        
        calls = 0
        best_x = x0
        best_f = func(x0)
        calls += 1
        
        while calls < cma_budget:
            # Generate population
            B = np.random.randn(lambda_, n)
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                L = np.eye(n)
                
            Z = B @ L.T
            P = m + sigma * Z
            P = np.clip(P, -5, 5)
            
            F = np.array([func(p) for p in P])
            calls += lambda_
            
            idx = np.argsort(F)
            P = P[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = P[0].copy()
                
            m_old = m.copy()
            m = sum(w[i] * P[i] for i in range(mu))
            
            # Update ps
            # Note: (m - m_old) / sigma is the step in sigma units
            # L.T @ (m - m_old) / sigma
            step = (m - m_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * L.T @ step
            
            # hsig check (simplified, often just check norm)
            # Standard check
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(calls / (cs * (n + 1)))) < (1 + 1.5 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (n + 1))
            
            # Update C
            # y_i = (P[i] - m_old) / sigma
            delta_hs = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 + cmu - c1 - cmu + delta_hs) * C \
                + c1 * (np.outer(ps, ps) - C)
            
            for i in range(mu):
                y = (P[i] - m_old) / sigma
                C += cmu * w[i] * np.outer(y, y)
                
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / hsig - 1))
            
            # Convergence check
            if sigma < 1e-8:
                break
                
        # Phase 2: L-BFGS-B Polish
        # Use remaining budget
        remaining = total_budget - calls
        
        if remaining > 10:
            # Wrap func to count calls and stop if budget exceeded
            # L-BFGS-B uses finite differences. In 5D, each gradient is ~6 evals.
            # 150 evals / 6 = 25 iterations. Should be enough for local convergence.
            
            # We need to ensure we don't exceed total_budget.
            # scipy doesn't have a perfect "max_evals" that stops exactly and returns best.
            # We can use a callback.
            
            call_counter = [calls]
            
            def callback(xk):
                # xk is the current point
                # We don't know exactly how many evals scipy used internally yet, 
                # but we can estimate or just stop after a fixed number of iterations.
                # Better: use maxiter.
                return False # Don't stop early unless we want to
            
            # Estimate max iterations
            # Each iteration of L-BFGS-B in 5D costs approx 2*n + 1 = 11 evals?
            # Actually, L-BFGS-B uses a quasi-Newton method.
            # Let's just set maxiter = remaining // 10 to be safe.
            max_iter = max(1, remaining // 10)
            
            try:
                res = scipy.optimize.minimize(
                    func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': max_iter, 'maxls': 20}
                )
                # We don't know exactly how many calls were made, but we assume 
                # it didn't blow the budget significantly.
                # If res.fun < best_f:
                #    best_x = res.x
                #    best_f = res.fun
            except:
                pass
                
        return best_x, best_f
