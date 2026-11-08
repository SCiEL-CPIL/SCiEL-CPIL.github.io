import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Optimize discrete heights directly using L-BFGS-B with
    multiple random restarts and structural priors (sparse clusters).
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    def compute_ratio(h):
        """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-15:
            return 0.0
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    from scipy.optimize import minimize
    
    def neg_ratio(h):
        h = np.maximum(0, h)
        return -compute_ratio(h)
    
    best_ratio = 0.0
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_ratio = compute_ratio(best_h)
    
    # Strategy 1: Optimize small sparse clusters directly
    # Try clusters of size k=1,2,3,4,5 at various positions
    for k in range(1, 6):
        if time.time() > deadline - 10.0:
            break
        # Try different starting positions
        for start in range(0, N - k + 1, max(1, (N - k) // 20)):
            if time.time() > deadline - 10.0:
                break
            # Initial: uniform on cluster
            h0 = np.zeros(N)
            h0[start:start+k] = 1.0/k
            
            # Optimize the k values in the cluster
            def sub_obj(params):
                h = np.zeros(N)
                h[start:start+k] = np.maximum(0, params)
                return -compute_ratio(h)
            
            try:
                res = minimize(sub_obj, h0[start:start+k].copy(), 
                             method='L-BFGS-B', 
                             bounds=[(0, None)]*k,
                             options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < -best_ratio:
                    h_cand = np.zeros(N)
                    h_cand[start:start+k] = np.maximum(0, res.x)
                    r = compute_ratio(h_cand)
                    if r > best_ratio:
                        best_ratio = r
                        best_h = h_cand.copy()
            except Exception:
                continue
    
    # Strategy 2: Full optimization with sparsity-inducing initialization
    # Use several random sparse initializations
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 15.0:
            break
        
        # Random sparse initialization
        n_nonzero = np.random.randint(1, 10)
        indices = np.random.choice(N, size=n_nonzero, replace=False)
        h0 = np.zeros(N)
        h0[indices] = np.random.uniform(0.1, 2.0, size=n_nonzero)
        
        try:
            res = minimize(neg_ratio, h0.copy(), 
                         method='L-BFGS-B',
                         bounds=[(0, None)]*N,
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            h_cand = np.maximum(0, res.x)
            r = compute_ratio(h_cand)
            if r > best_ratio:
                best_ratio = r
                best_h = h_cand.copy()
        except Exception:
            continue
    
    # Strategy 3: Polish the best solution with tighter tolerances
    if time.time() < deadline - 5.0 and best_ratio > 0:
        try:
            res = minimize(neg_ratio, best_h.copy(), 
                         method='L-BFGS-B',
                         bounds=[(0, None)]*N,
                         options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            h_cand = np.maximum(0, res.x)
            r = compute_ratio(h_cand)
            if r > best_ratio:
                best_ratio = r
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Final: ensure non-negativity and validity
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h[0] = 1.0
    
    return list(best_h)
# EVOLVE-BLOCK-END
