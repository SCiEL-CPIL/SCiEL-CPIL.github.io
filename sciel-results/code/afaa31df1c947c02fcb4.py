import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # We use a Fourier series parameterization:
    # h_i = 1/N * (1 + sum_{k=1}^K a_k cos(2*pi*k*i/N))
    # We enforce non-negativity by bounding the amplitudes such that the sum of absolute values < 1.
    # To be safe, we can use a scaling factor or just bound each a_k.
    # Let's use K=10 harmonics.
    K = 10
    n_params = K
    
    # Precompute cosine bases
    i = np.arange(N)
    cos_bases = np.array([np.cos(2 * np.pi * k * i / N) for k in range(1, K+1)])
    
    def get_p(params):
        # params: [a1, ..., aK]
        # p = 1/N * (1 + sum a_k cos_k)
        # We ensure non-negativity by clipping, but we should constrain params to avoid clipping.
        # If we bound |a_k| <= 1/(2K), then sum |a_k| <= 1/2 < 1, so 1 + sum ... >= 1/2 > 0.
        p = 1.0 / N * (1.0 + cos_bases @ params)
        # Just in case, clip and renormalize
        p = np.clip(p, 0, None)
        s = np.sum(p)
        if s < 1e-10:
            p = np.ones(N) / N
        else:
            p = p / s
        return p

    def objective(params):
        p = get_p(params)
        c = np.convolve(p, p, mode='full')
        return c.max()
    
    # Initial guess: uniform (all params 0)
    best_p = np.ones(N) / N
    best_val = np.convolve(best_p, best_p, mode='full').max()
    
    # Define bounds for L-BFGS-B
    # If we set bounds to [-0.1, 0.1] for each, max deviation is 10 * 0.1 = 1.0.
    # This allows p to go to 0.
    # Let's use tighter bounds to stay in the "smooth" region.
    bounds = [(-0.05, 0.05) for _ in range(K)]
    
    # Try multiple random starts
    np.random.seed(42)
    starts = [np.zeros(K)]
    for _ in range(10):
        starts.append(np.random.uniform(-0.02, 0.02, K))
        
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_p = get_p(res.x)
                best_p = np.clip(best_p, 0, None)
                s = np.sum(best_p)
                if s > 1e-10:
                    best_p = best_p / s
        except Exception:
            continue
            
    return [float(x) for x in best_p]
