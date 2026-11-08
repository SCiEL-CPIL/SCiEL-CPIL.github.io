"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    
    Strategy: Use numerical optimization to find better configurations.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    def ratio(h):
        h = np.array(h, dtype=np.float64)
        if np.any(h < 0):
            return 0.0
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf == 0:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    # Try several initial configurations and optimize
    best_h = None
    best_ratio = 0.0
    
    # Initial configurations to try
    inits = []
    
    # 1. 2-point cluster 2:1 (current best: 1.1944)
    h1 = [0.0] * N
    h1[0] = 2.0 / 3.0
    h1[1] = 1.0 / 3.0
    inits.append(h1)
    
    # 2. Single spike
    h2 = [0.0] * N
    h2[0] = 1.0
    inits.append(h2)
    
    # 3. Uniform
    inits.append([1.0] * N)
    
    # 4. 3-point cluster with varying ratios
    for ratio_3 in [0.5, 1.0, 2.0, 3.0]:
        h3 = [0.0] * N
        total = 1.0 + ratio_3 + 1.0
        h3[0] = 1.0 / total
        h3[1] = ratio_3 / total
        h3[2] = 1.0 / total
        inits.append(h3)
    
    # 5. Spread out configurations
    for spread in [2, 3, 5, 10]:
        h5 = [0.0] * N
        for i in range(0, N, spread):
            h5[i] = 1.0 / (N // spread + 1)
        inits.append(h5)
    
    # 6. Gaussian-like
    h6 = [0.0] * N
    center = N // 2
    for i in range(N):
        h6[i] = np.exp(-((i - center) ** 2) / (2 * (N/4) ** 2))
    h6 = [x / sum(h6) for x in h6]
    inits.append(h6)
    
    # Evaluate initial configurations
    for h in inits:
        r = ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h[:]
    
    # Use scipy.optimize to improve the best configuration
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(x):
            return -ratio(x)
        
        # Optimize from multiple starting points
        for h0 in inits[:10]:  # Limit to 10 starting points for time
            if time.time() > deadline - 30:  # Safety margin
                break
                
            try:
                result = minimize(neg_ratio, h0, method='L-BFGS-B', 
                                 bounds=[(0, None)] * N,
                                 options={'maxiter': 1000, 'ftol': 1e-10})
                if result.fun < -best_ratio:
                    best_ratio = -result.fun
                    best_h = result.x.tolist()
            except Exception:
                pass
    except ImportError:
        pass
    
    # If no improvement found, return the best initial
    if best_h is None:
        return [1.0] * N
    
    return best_h
# EVOLVE-BLOCK-END
