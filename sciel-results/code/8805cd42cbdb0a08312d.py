import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for a given height array."""
    h = np.array(h, dtype=np.float64)
    if np.sum(h) == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    linf = np.max(conv)
    if l1 * linf == 0:
        return 0.0
    return l2_sq / (l1 * linf)

def neg_ratio(h):
    """Negative of ratio for minimization."""
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_score = 0.0
    
    # 1. Try the known best simple case: single spike
    h_spikes = np.zeros(N)
    h_spikes[0] = 1.0
    best_score = compute_ratio(h_spikes)
    best_h = h_spikes.copy()
    
    # 2. Parametric Optimization of 2-point clusters
    # Parameters: [pos1, pos2, weight_ratio]
    # We fix the total mass to 1 for simplicity, or just optimize weights directly.
    # Let's optimize weights w1, w2 and positions p1, p2.
    # To keep it simple and robust, we iterate over discrete positions and optimize weights.
    
    def optimize_weights_for_positions(pos_list, init_weights=None):
        if len(pos_list) == 0:
            return 0.0, np.zeros(N)
        
        if init_weights is None:
            init_weights = np.ones(len(pos_list)) / len(pos_list)
        
        def obj(w):
            h = np.zeros(N)
            for i, pos in enumerate(pos_list):
                h[pos] = w[i]
            # Ensure non-negative
            h = np.maximum(h, 0)
            if np.sum(h) == 0:
                return 0.0
            return -compute_ratio(h)
        
        try:
            res = minimize(obj, init_weights, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            w_opt = np.maximum(res.x, 0)
            h_opt = np.zeros(N)
            for i, pos in enumerate(pos_list):
                h_opt[pos] = w_opt[i]
            return compute_ratio(h_opt), h_opt
        except Exception:
            return 0.0, np.zeros(N)

    # Iterate over pairs of positions
    # To save time, we don't check all 1225 pairs. We check pairs with small separation and some larger ones.
    # Also, the problem is translation invariant? No, boundaries matter.
    # But for 2 points, the relative position and distance from boundary matter.
    
    # Let's try a grid of positions
    for p1 in range(N):
        if time.time() > deadline - 5:
            break
        for p2 in range(p1, N):
            if p2 - p1 > 10 and (p1 > 5 or p2 < N-5):
                continue # Skip very far apart or interior pairs if we are running out of time? 
                # Actually, let's just be smart about it.
            
            # Only optimize if the pair is "interesting" or we have time
            if time.time() > deadline - 2:
                continue
                
            # Initial guess: equal weights
            score, h_opt = optimize_weights_for_positions([p1, p2], [0.5, 0.5])
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
    # 3. Try 3-point clusters with small separation
    if time.time() < deadline - 10:
        for p1 in range(N-2):
            if time.time() > deadline - 5:
                break
            for sep1 in [1, 2]:
                for sep2 in [1, 2]:
                    if time.time() > deadline - 2:
                        continue
                    p2 = p1 + sep1
                    p3 = p2 + sep2
                    if p3 >= N:
                        continue
                    score, h_opt = optimize_weights_for_positions([p1, p2, p3], [1/3, 1/3, 1/3])
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
    
    # 4. Random Sparse Refinement
    # Pick random small subsets and optimize
    rng = np.random.RandomState(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        k = rng.randint(2, 5)
        pos = rng.choice(N, k, replace=False)
        pos = sorted(pos)
        score, h_opt = optimize_weights_for_positions(pos, np.ones(k)/k)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # 5. Final polish: Nelder-Mead on the full vector starting from best_h
    # This can help if the best_h is a local minimum in the sparse subspace but can be improved by small non-zero neighbors
    if time.time() < deadline - 10:
        # Add small noise to allow escape from sparse subspace if beneficial
        h_start = best_h + 0.01 * rng.random(N)
        h_start = np.maximum(h_start, 0)
        
        try:
            res = minimize(neg_ratio, h_start, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 0:
                score = compute_ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    return list(best_h)
