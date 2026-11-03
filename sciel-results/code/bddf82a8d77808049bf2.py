import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N
_NUM_LAGS = 2 * N - 1

# Precompute overlap matrices
def _build_overlap_matrices():
    M = np.zeros((_NUM_LAGS, N, N))
    for k in range(_NUM_LAGS):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()

def _compute_ratio_full(h):
    """Compute ratio for full 50-element vector."""
    h = np.maximum(h, 0)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    sup_ff = np.max(ff_vals)
    if sup_ff < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_ff)

def _compute_ratio_sparse(weights, support):
    """Compute ratio for sparse representation: weights on support positions."""
    if len(weights) == 0 or len(support) == 0:
        return 0.0
    h = np.zeros(N)
    for i, pos in enumerate(support):
        h[pos] = weights[i]
    return _compute_ratio_full(h)

def _optimize_weights_for_support(support, max_iter=200, tol=1e-10):
    """Optimize weights for a fixed support using multiplicative updates + coordinate descent."""
    K = len(support)
    if K == 0:
        return np.array([])
    
    # Initialize weights uniformly
    w = np.ones(K) / K
    
    # Build sub-matrices for this support
    # For each lag k, the contribution is sum_{a,b} w[a]*w[b]*M[k, support[a], support[b]]
    M_sub = np.zeros((_NUM_LAGS, K, K))
    for k in range(_NUM_LAGS):
        for a in range(K):
            for b in range(K):
                M_sub[k, a, b] = _M_matrices[k, support[a], support[b]]
    
    def _ratio_from_w(w):
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s < 1e-15:
            return 0.0
        w_norm = w / s
        q = np.sum(w_norm**2)
        if q < 1e-15:
            return 0.0
        ff_vals = np.array([DELTA * (w_norm @ M_sub[k] @ w_norm) for k in range(_NUM_LAGS)])
        sup_ff = np.max(ff_vals)
        if sup_ff < 1e-15:
            return 0.0
        # Ratio = q / sup_ff (since s=1 after normalization)
        return q / sup_ff
    
    # Multiplicative updates
    for iteration in range(max_iter):
        w = np.maximum(w, 1e-15)
        s = np.sum(w)
        w_norm = w / s
        
        # Compute Q_k = w_norm @ M_sub[k] @ w_norm
        Q = np.array([w_norm @ M_sub[k] @ w_norm for k in range(_NUM_LAGS)])
        
        # Softmax over lags with temperature
        temperature = 0.05
        max_Q = np.max(Q)
        if max_Q < 1e-15:
            break
        p = np.exp((Q - max_Q) / temperature)
        p_sum = np.sum(p)
        if p_sum < 1e-15:
            break
        p = p / p_sum
        
        # Effective matrix
        M_eff = np.sum([p[k] * M_sub[k] for k in range(_NUM_LAGS)], axis=0)
        
        # Multiplicative update: minimize w @ M_eff @ w
        Mh = M_eff @ w_norm
        Mh = np.maximum(Mh, 1e-15)
        w_new = w_norm / np.sqrt(Mh)
        s_new = np.sum(w_new)
        if s_new < 1e-15:
            break
        w_new = w_new / s_new
        
        if np.max(np.abs(w_new - w_norm)) < tol:
            w = w_new
            break
        w = w_new
    
    # Coordinate descent polish
    for iteration in range(100):
        improved = False
        for i in range(K):
            w_old_i = w[i]
            best_val = w_old_i
            best_r = _ratio_from_w(w)
            
            # Golden section search
            lo = 0.0
            hi = max(1.0, 3.0 * w_old_i) if w_old_i > 0 else 1.0
            gr = (np.sqrt(5) - 1) / 2
            
            for _ in range(15):
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                
                w[i] = x1
                r1 = _ratio_from_w(w)
                w[i] = x2
                r2 = _ratio_from_w(w)
                
                if r1 > r2:
                    hi = x2
                else:
                    lo = x1
            
            best_h_i = (lo + hi) / 2
            w[i] = best_h_i
            r_new = _ratio_from_w(w)
            
            if r_new > best_r + 1e-12:
                improved = True
            else:
                w[i] = w_old_i
        
        if not improved:
            break
    
    return w

def _generate_support_candidates():
    """Generate diverse support patterns to try."""
    candidates = []
    
    # Sparse patterns: K = 2 to 12
    # For each K, generate several random supports and some structured ones
    
    # Structured: evenly spaced
    for K in range(2, 13):
        if K > N:
            continue
        # Evenly spaced
        spacing = N // K
        support = list(range(0, N, spacing))[:K]
        if len(support) == K:
            candidates.append(tuple(support))
        
        # Offset evenly spaced
        for offset in range(min(spacing, 3)):
            support = list(range(offset, N, spacing))[:K]
            if len(support) == K:
                candidates.append(tuple(support))
    
    # Random supports
    np.random.seed(42)
    for K in range(2, 13):
        for trial in range(5):
            support = tuple(np.random.choice(N, K, replace=False).tolist())
            candidates.append(support)
    
    # Sidon-like sets (minimize collisions in sums)
    def _sidon_set(n, size):
        """Generate a Sidon set of given size in [0, n)."""
        seq = [0]
        used_sums = {0}
        for candidate in range(1, n):
            valid = True
            for i in seq:
                if i + candidate in used_sums:
                    valid = False
                    break
            if valid:
                seq.append(candidate)
                used_sums.add(candidate + candidate)
                for i in seq[:-1]:
                    used_sums.add(i + candidate)
            if len(seq) >= size:
                break
        return seq[:size]
    
    for K in range(2, 13):
        sidon = _sidon_set(N, K)
        if len(sidon) == K:
            candidates.append(tuple(sidon))
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try all support candidates
    supports = _generate_support_candidates()
    
    for support in supports:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        K = len(support)
        if K < 2 or K > 12:
            continue
        
        # Optimize weights for this support
        weights = _optimize_weights_for_support(support, max_iter=150, tol=1e-9)
        
        if len(weights) == 0:
            continue
        
        # Build full vector and compute score
        h = np.zeros(N)
        for i, pos in enumerate(support):
            h[pos] = weights[i]
        
        score = _compute_ratio_full(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Final polish with coordinate descent on the full vector
    if time.time() - start_time < DEADLINE - 10 and best_score > 0:
        try:
            h_polished = best_h.copy()
            for iteration in range(500):
                if time.time() - start_time > DEADLINE - 5:
                    break
                improved = False
                for i in range(N):
                    h_old_i = h_polished[i]
                    best_r = _compute_ratio_full(h_polished)
                    
                    # Quick check: if h[i] is zero, try setting it to small value
                    if h_old_i < 1e-10:
                        h_polished[i] = 0.01
                        r_new = _compute_ratio_full(h_polished)
                        if r_new > best_r + 1e-10:
                            improved = True
                        else:
                            h_polished[i] = 0.0
                        continue
                    
                    # Golden section search
                    lo = 0.0
                    hi = max(1.0, 3.0 * h_old_i)
                    gr = (np.sqrt(5) - 1) / 2
                    
                    for _ in range(12):
                        x1 = hi - gr * (hi - lo)
                        x2 = lo + gr * (hi - lo)
                        
                        h_polished[i] = x1
                        r1 = _compute_ratio_full(h_polished)
                        h_polished[i] = x2
                        r2 = _compute_ratio_full(h_polished)
                        
                        if r1 > r2:
                            hi = x2
                        else:
                            lo = x1
                    
                    best_h_i = (lo + hi) / 2
                    h_polished[i] = best_h_i
                    r_new = _compute_ratio_full(h_polished)
                    
                    if r_new > best_r + 1e-12:
                        improved = True
                    else:
                        h_polished[i] = h_old_i
                
                if not improved:
                    break
            
            final_score = _compute_ratio_full(h_polished)
            if final_score > best_score:
                best_score = final_score
                best_h = h_polished.copy()
        except Exception:
            pass
    
    return best_h.tolist()
