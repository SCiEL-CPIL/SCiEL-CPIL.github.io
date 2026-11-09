import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution():
        start_time = time.time()
        deadline = 600.0
        
        n = 10
        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x, y, indexing='ij') # X[i,j] = i, Y[i,j] = j
        # Wait, P[y, x]. So P[j, i].
        # Let's stick to P[y, x] where y is row, x is col.
        # X values are cols 0..9. Y values are rows 0..9.
        
        # Centered coords
        Xc = X - 4.5
        Yc = Y - 4.5
        
        # Projections
        V_m1 = X - Y
        V_0 = X
        V_1 = X + Y
        V_2 = X + 2*Y
        
        def get_entropy(vals, probs):
            # vals: 10x10 array of keys
            # probs: 10x10 array of weights
            unique_vals, inverse = np.unique(vals, return_inverse=True)
            # Sum probs for each unique val
            counts = np.bincount(inverse, weights=probs)
            # Filter zeros
            counts = counts[counts > 1e-12]
            if len(counts) == 0:
                return 0.0
            return -np.sum(counts * np.log2(counts))
        
        def objective(params):
            alpha, beta, gamma = params
            # Log weights
            log_w = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
            # Numerical stability
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            total = np.sum(w)
            if total < 1e-12:
                return 1e6 # Bad
            P = w / total
            
            D = get_entropy(V_m1, P)
            H0 = get_entropy(V_0, P)
            H1 = get_entropy(V_1, P)
            H2 = get_entropy(V_2, P)
            M = max(H0, H1, H2)
            
            if M < 1e-9:
                return 1e6
            return -D / M
        
        best_score = -1.0
        best_P = np.ones((10,10)) / 100.0 # Uniform fallback
        
        # Initializations
        # 1. Uniform equivalent (alpha=beta=gamma=0)
        # 2. Anti-diagonal band (gamma negative large, alpha=beta small)
        # 3. Random
        
        inits = [
            [0.1, 0.1, 0.0],
            [0.5, 0.5, -0.5],
            [0.2, 0.2, -1.0],
            [1.0, 1.0, -1.0],
            [0.5, 0.5, 0.5],
            [2.0, 0.1, -0.2]
        ]
        
        # Add random restarts
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.uniform(0.01, 2.0)
            b = rng.uniform(0.01, 2.0)
            g = rng.uniform(-1.5, 1.5)
            inits.append([a, b, g])
        
        for init in inits:
            if time.time() > deadline - 30:
                break
                
            try:
                res = minimize(objective, init, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_score:
                    best_score = res.fun
                    alpha, beta, gamma = res.x
                    log_w = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
                    log_w -= np.max(log_w)
                    w = np.exp(log_w)
                    best_P = w / np.sum(w)
            except Exception:
                pass
        
        # Final check and cleanup
        best_P = np.clip(best_P, 0, None)
        s = np.sum(best_P)
        if s > 1e-9:
            best_P = best_P / s
        
        return best_P
