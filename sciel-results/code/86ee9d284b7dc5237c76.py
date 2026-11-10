import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Time budget guard
    start_time = time.time()
    timeout = 600.0
    
    def compute_entropies(P):
        # P is (10, 10), P[y, x]
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-9:
            return 0.0, 0.0, 0.0, 0.0
        P = P / s
        
        # H_0: entropy of X
        px = P.sum(axis=0) # sum over y for each x
        px = px[px > 0]
        H0 = -np.sum(px * np.log2(px))
        
        # H_1: entropy of X + Y
        # Z = x + y
        # x in 0..9, y in 0..9 => Z in 0..18
        pzy = np.zeros(19)
        for y in range(n):
            for x in range(n):
                z = x + y
                pzy[z] += P[y, x]
        pzy = pzy[pzy > 0]
        H1 = -np.sum(pzy * np.log2(pzy))
        
        # H_2: entropy of X + 2Y
        # Z = x + 2y
        # x in 0..9, y in 0..9 => Z in 0..27
        pzy2 = np.zeros(28)
        for y in range(n):
            for x in range(n):
                z = x + 2 * y
                pzy2[z] += P[y, x]
        pzy2 = pzy2[pzy2 > 0]
        H2 = -np.sum(pzy2 * np.log2(pzy2))
        
        # H_-1: entropy of X - Y
        # Z = x - y
        # x in 0..9, y in 0..9 => Z in -9..9
        # Shift by 9 to index 0..18
        pzy_minus = np.zeros(19)
        for y in range(n):
            for x in range(n):
                z = x - y + 9
                pzy_minus[z] += P[y, x]
        pzy_minus = pzy_minus[pzy_minus > 0]
        Hm1 = -np.sum(pzy_minus * np.log2(pzy_minus))
        
        return H0, H1, H2, Hm1

    def objective(params):
        # params are flattened U and V
        r = 4 # rank
        U = params[:n*r].reshape(n, r)
        V = params[n*r:].reshape(n, r)
        P = U @ V.T
        # Add penalty for negative values to steer towards valid distributions
        neg_penalty = np.sum(np.clip(-P, 0, None) ** 2) * 0.1
        
        H0, H1, H2, Hm1 = compute_entropies(P)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 1e6 + neg_penalty
        ratio = Hm1 / M
        return -ratio + neg_penalty

    best_P = np.ones((n, n)) / (n*n)
    best_score = 1.0
    
    # Try a few random initializations
    for i in range(5):
        if time.time() - start_time > timeout - 60:
            break
        
        np.random.seed(42 + i)
        # Initialize U and V with small random values
        U = np.random.randn(n, 4) * 0.1
        V = np.random.randn(n, 4) * 0.1
        params0 = np.concatenate([U.flatten(), V.flatten()])
        
        res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
        
        if res.success or res.fun < -best_score + 1e-6:
            U_opt = res.x[:n*4].reshape(n, 4)
            V_opt = res.x[n*4:].reshape(n, 4)
            P_opt = U_opt @ V_opt.T
            P_opt = np.clip(P_opt, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                H0, H1, H2, Hm1 = compute_entropies(P_opt)
                M = max(H0, H1, H2)
                if M > 1e-9:
                    score = Hm1 / M
                    if score > best_score:
                        best_score = score
                        best_P = P_opt

    return best_P
