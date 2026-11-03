import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    def evaluate_ratio(h):
        """
        Computes the ratio ||h||_2^2 / (||h||_1 * sup(h*h))
        h is a numpy array of length N
        """
        h = np.array(h, dtype=float)
        s1 = np.sum(h)
        if s1 < 1e-12:
            return 0.0
        s2 = np.sum(h**2)
        # Linear convolution
        conv = np.convolve(h, h, mode='full')
        m = np.max(conv)
        if m < 1e-12:
            return 0.0
        return s2 / (s1 * m)

    def objective(h):
        """
        Objective for minimization: -log(ratio) or -ratio
        We want to maximize ratio.
        Fix sum(h) = 1 to avoid scale issues?
        If we don't fix sum, the optimizer might scale h to infinity?
        Ratio is scale invariant.
        Let's fix sum(h)=1 by normalizing inside?
        No, if we normalize inside, the gradient is 0.
        Better: Optimize over h with sum(h)=1 constraint?
        Or just optimize h, and the ratio is invariant.
        But Nelder-Mead might wander.
        Let's normalize h to sum 1 in the evaluation, but pass raw h to optimizer?
        If we normalize, the effective dimension is K-1.
        
        Let's stick to optimizing the weights on the support, normalized to sum 1.
        """
        # h is the vector of weights on the support
        if np.any(h < 0):
            return 1e10
        s = np.sum(h)
        if s < 1e-12:
            return 1e10
        h_norm = h / s
        # Create full vector
        full_h = np.zeros(N)
        full_h[support_indices] = h_norm
        ratio = evaluate_ratio(full_h)
        # Minimize -ratio
        return -ratio

    # Phase 1: Find good supports
    best_supports = []
    best_ratio = 0.0
    
    # Try different K
    for K in range(4, 16):
        if time.time() > deadline - 10:
            break
        num_samples = 200
        # Reduce samples if time is tight
        if time.time() > deadline - 30:
            num_samples = 50
            
        for _ in range(num_samples):
            if time.time() > deadline - 10:
                break
            # Random subset of size K
            indices = np.random.choice(N, K, replace=False)
            indices.sort()
            # Uniform weights
            h = np.zeros(N)
            h[indices] = 1.0 / K
            ratio = evaluate_ratio(h)
            if ratio > best_ratio:
                best_ratio = ratio
                # Keep track of top supports
                if len(best_supports) < 10:
                    best_supports.append((ratio, indices))
                    best_supports.sort(key=lambda x: x[0], reverse=True)
                else:
                    if ratio > best_supports[-1][0]:
                        best_supports[-1] = (ratio, indices)
                        best_supports.sort(key=lambda x: x[0], reverse=True)

    if not best_supports:
        # Fallback
        return [1.0] + [0.0] * (N-1)

    # Phase 2: Optimize weights on the best supports
    global_best_h = None
    global_best_ratio = best_ratio
    
    for ratio, indices in best_supports[:5]: # Top 5
        if time.time() > deadline - 5:
            break
            
        K = len(indices)
        x0 = np.ones(K) / K
        
        # Bounds: h_i >= 0
        bounds = [(0, 1)] * K
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           bounds=bounds, 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < -global_best_ratio:
                # res.fun is -ratio
                new_ratio = -res.fun
                if new_ratio > global_best_ratio:
                    global_best_ratio = new_ratio
                    h_opt = res.x
                    s = np.sum(h_opt)
                    if s > 0:
                        h_opt = h_opt / s
                    full_h = np.zeros(N)
                    full_h[indices] = h_opt
                    global_best_h = full_h
        except Exception:
            pass

    if global_best_h is None:
        # Use the best uniform support
        ratio, indices = best_supports[0]
        h = np.zeros(N)
        h[indices] = 1.0 / len(indices)
        global_best_h = h

    # Ensure non-negative and list format
    return list(global_best_h)
# EVOLVE-BLOCK-END
