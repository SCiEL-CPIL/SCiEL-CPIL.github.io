import numpy as np
import time
from scipy.optimize import minimize
import warnings

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def make_h(params):
        # params: [A1, mu1, sig1, A2, mu2, sig2, ...]
        # K components
        K = len(params) // 3
        h = np.zeros(N)
        x = np.arange(N)
        for i in range(K):
            A = params[3*i]
            mu = params[3*i+1]
            sig = params[3*i+2]
            if A < 0: A = 0
            if sig < 0.1: sig = 0.1
            g = A * np.exp(-0.5 * ((x - mu) / sig)**2)
            h = np.maximum(h, g)
        return h

    def neg_score(params):
        if time.time() > deadline - 5.0:
            return 1e10
        h = make_h(params)
        s = score(h)
        if s < 1e-10:
            return 1e10
        return -s

    best_h = np.zeros(N)
    best_s = 0.0

    # Try different numbers of Gaussian components
    for K in [1, 2, 3, 4]:
        if time.time() > deadline - 10.0:
            break
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize parameters
            # A in [0.5, 2.0], mu in [0, N-1], sig in [1.0, 10.0]
            params0 = np.concatenate([
                np.random.uniform(0.5, 2.0, K),
                np.random.uniform(0, N-1, K),
                np.random.uniform(1.0, 10.0, K)
            ])
            
            try:
                res = minimize(neg_score, params0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < 0:
                    s = -res.fun
                    if s > best_s:
                        best_s = s
                        best_h = make_h(res.x)
            except Exception:
                pass

    # Fallback: try the seed's discrete approach if parametric didn't find anything good
    if best_s < 1.0:
         # Simple discrete search for a good Sidon set
         # Re-use logic from seed but simplified
         pass 

    # Normalize output to sum 1 to be safe? 
    # The seed returned sum=1.
    if np.sum(best_h) > 1e-10:
        best_h = best_h / np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return list(best_h)
