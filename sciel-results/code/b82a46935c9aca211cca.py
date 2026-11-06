import numpy as np
import time

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    N_COEFF = 24
    sqrt_N = np.sqrt(N_COEFF)
    
    N_GRID = 2048
    theta = 2 * np.pi * np.arange(N_GRID) / N_GRID
    z = np.exp(1j * theta)
    Z = z[None, :] ** np.arange(N_COEFF)[:, None]
    
    KC = np.arange(N_COEFF)
    
    def score_fast(coeffs):
        vals = coeffs @ Z
        return np.max(np.abs(vals))
    
    def score_precise(coeffs, top_k=10, iters=50):
        vals = coeffs @ Z
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = m
        
        gr = (np.sqrt(5) - 1) / 2
        
        lo_idx = (idx - 1) % N_GRID
        hi_idx = (idx + 1) % N_GRID
        a = theta[lo_idx].copy()
        b = theta[hi_idx].copy()
        b[a > b] += 2 * np.pi
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        for _ in range(iters):
            zc = np.exp(1j * c)
            zd = np.exp(1j * d)
            pc = np.array([np.dot(coeffs, zc[i] ** KC) for i in range(top_k)])
            pd = np.array([np.dot(coeffs, zd[i] ** KC) for i in range(top_k)])
            ac = np.abs(pc)
            ad = np.abs(pd)
            
            keep_c = ac < ad
            
            a[~keep_c] = c[~keep_c]
            b[keep_c] = d[keep_c]
            
            c_new = np.empty(top_k)
            d_new = np.empty(top_k)
            c_new[keep_c] = b[keep_c] - gr * (b[keep_c] - a[keep_c])
            d_new[keep_c] = c[keep_c]
            c_new[~keep_c] = d[~keep_c]
            d_new[~keep_c] = a[~keep_c] + gr * (b[~keep_c] - a[~keep_c])
            
            c = c_new
            d = d_new
            
            if time.time() > deadline - 5:
                break
        
        zm = np.exp(1j * (a + b) / 2)
        pm = np.array([np.dot(coeffs, zm[i] ** KC) for i in range(top_k)])
        am = np.abs(pm)
        best = max(best, np.max(am))
        
        return best

    # Exhaustive search: fix c[0]=1 (sign symmetry), enumerate 2^23
    best_score = float('inf')
    best_coeffs = None
    
    BATCH_SIZE = 16384
    total = 1 << 23  # 8388608
    
    for batch_start in range(0, total, BATCH_SIZE):
        if time.time() > deadline - 30:
            break
        
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_size = batch_end - batch_start
        
        indices = np.arange(batch_start, batch_end, dtype=np.int64)
        
        coeff_mat = np.ones((batch_size, N_COEFF), dtype=np.float64)
        for bit in range(23):
            mask = (indices >> bit) & 1
            coeff_mat[mask == 1, 1 + bit] = -1.0
        
        # Compute max |P(z)| for each coefficient vector in sub-batches
        SUB_BATCH = 4096
        max_abs_all = np.empty(batch_size, dtype=np.float64)
        
        for sub_start in range(0, batch_size, SUB_BATCH):
            sub_end = min(sub_start + SUB_BATCH, batch_size)
            sub_coeffs = coeff_mat[sub_start:sub_end]
            P_vals = sub_coeffs @ Z
            max_abs_all[sub_start:sub_end] = np.max(np.abs(P_vals), axis=1)
            del P_vals
        
        batch_best_idx = np.argmin(max_abs_all)
        batch_best_score = max_abs_all[batch_best_idx]
        
        if batch_best_score < best_score:
            best_score = batch_best_score
            best_coeffs = coeff_mat[batch_best_idx].copy()
            
            if best_score <= 1.2583758 * sqrt_N + 1e-5:
                break
    
    # Precise refinement of the best found
    if best_coeffs is not None and time.time() < deadline - 5:
        final_score = score_precise(best_coeffs, top_k=16, iters=80)
        if final_score < best_score:
            best_score = final_score
    
    if best_coeffs is None:
        # Fallback: Golay-Rudin-Shapiro
        p, q = [1], [1]
        while len(p) < N_COEFF:
            p, q = p + q, p + [-c for c in q]
        best_coeffs = np.array(p[:N_COEFF], dtype=np.float64)
    
    return [int(c) for c in best_coeffs]
