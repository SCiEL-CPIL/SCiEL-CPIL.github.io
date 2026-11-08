import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start = time.time()
    deadline = start + 110  # safety margin
    
    # Compute autoconvolution matrix: (h*h)[k] = sum_i h[i]*h[k-i]
    # k ranges from 0 to 2*(N-1) = 98
    # Build the linear operator for autoconvolution
    
    def autoconv(h):
        """Compute autoconvolution of h (length N), returns array of length 2N-1."""
        result = np.zeros(2*N-1)
        for k in range(2*N-1):
            s = 0.0
            for i in range(max(0, k-(N-1)), min(N-1, k)+1):
                j = k - i
                s += h[i] * h[j]
            result[k] = s
        return result
    
    # Better: use FFT or direct convolution
    def autoconv_fast(h):
        # Use np.convolve for autoconvolution
        return np.convolve(h, h)
    
    def objective_and_grad(w):
        """Minimize max_k (h*h)[k] with soft approximation, subject to sum(w)=1, w>=0."""
        conv = np.convolve(w, w)
        # Soft max approximation
        alpha = 100.0
        log_max = np.log(np.sum(np.exp(alpha * conv))) / alpha
        l2_sq = np.sum(w**2)
        # We want to maximize l2_sq / max_conv, equivalently minimize max_conv / l2_sq
        # Or minimize max_conv subject to l2_sq >= some threshold
        return log_max
    
    # Strategy: parametrize with a few active weights, optimize their values and positions
    # Use a subset of positions (Sidon-like) with optimized weights
    
    # First, find a good Sidon set in {0,...,49}
    # Greedy Sidon set construction
    def greedy_sidon(n, max_val):
        S = []
        used_sums = set()
        for v in range(max_val+1):
            if len(S) >= n:
                break
            # Check if v can be added
            ok = True
            for s in S:
                if v + s in used_sums:
                    ok = False
                    break
            if ok:
                # Also check v+v
                if v + v in used_sums:
                    ok = False
            if ok:
                S.append(v)
                for s in S:
                    used_sums.add(v + s)
        return S
    
    # Try different strategies and pick the best
    
    best_score = 0.0
    best_h = np.ones(N) / N  # default
    
    def score(h):
        h = np.array(h, dtype=float)
        if h.sum() < 1e-15:
            return 0.0
        h = h / h.sum()
        conv = np.convolve(h, h)
        l2_sq = np.sum(h**2)
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return l2_sq / max_conv
    
    # Strategy 1: Sidon set with equal weights
    for k in range(2, 12):
        S = greedy_sidon(k, N-1)
        if len(S) < k:
            continue
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0/k
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 2: Optimize weights on a Sidon set
    for k in range(3, 10):
        S = greedy_sidon(k, N-1)
        if len(S) < k:
            continue
        
        def obj(w_active):
            h = np.zeros(N)
            for i, idx in enumerate(S):
                h[idx] = w_active[i]
            h = h / h.sum()
            conv = np.convolve(h, h)
            l2_sq = np.sum(h**2)
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                return 1e10
            return -l2_sq / max_conv  # minimize negative
        
        x0 = np.ones(k) / k
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.success or res.fun < 0:
            w_opt = np.maximum(res.x, 0)
            h = np.zeros(N)
            for i, idx in enumerate(S):
                h[idx] = w_opt[i]
            s = score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Strategy 3: Exponential weights on all positions
    for base in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
        h = np.array([base**i for i in range(N)])
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 4: Two-sided exponential
    for base in [0.9, 0.8, 0.7, 0.6]:
        h = np.array([base**abs(i - N//2) for i in range(N)])
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 5: Optimize weights on all N positions using gradient-based method
    if time.time() < deadline - 20:
        # Use log-sum-exp for smooth max
        def smooth_obj(w):
            h = np.maximum(w, 0)
            h = h / h.sum()
            conv = np.convolve(h, h)
            l2_sq = np.sum(h**2)
            alpha = 200.0
            c_max = np.max(conv)
            if c_max < 1e-15:
                return 1e10
            log_max = c_max + np.log(np.sum(np.exp(alpha*(conv - c_max)))) / alpha
            return -l2_sq / log_max
        
        x0 = np.ones(N) / N
        res = minimize(smooth_obj, x0, method='L-BFGS-B', 
                       bounds=[(0, 1)]*N,
                       options={'maxiter': 200, 'ftol': 1e-14})
        if res.fun < 0:
            w_opt = np.maximum(res.x, 0)
            s = score(w_opt)
            if s > best_score:
                best_score = s
                best_h = w_opt.copy()
    
    # Strategy 6: Random restarts with local optimization on small subsets
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            # Pick random subset of size k
            k = rng.randint(3, 10)
            S = list(rng.choice(N, size=k, replace=False))
            
            def obj2(w_active):
                h = np.zeros(N)
                for i, idx in enumerate(S):
                    h[idx] = w_active[i]
                h = h / h.sum()
                conv = np.convolve(h, h)
                l2_sq = np.sum(h**2)
                max_conv = np.max(conv)
                if max_conv < 1e-15:
                    return 1e10
                return -l2_sq / max_conv
            
            x0 = np.ones(k) / k
            res = minimize(obj2, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < 0:
                w_opt = np.maximum(res.x, 0)
                h = np.zeros(N)
                for i, idx in enumerate(S):
                    h[idx] = w_opt[i]
                s = score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    # Final: return the best heights found
    best_h = np.maximum(best_h, 0)
    best_h = best_h / best_h.sum()
    return list(best_h)
