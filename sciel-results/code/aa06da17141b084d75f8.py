import numpy as np
import time
import scipy.optimize
import sys

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 < 2X4 - X3] (Wait, formula is X1+X2+X3 < 2X4).
    P = sum_{k,l} p_k p_l sum_{i,j: s_i+s_j < 2s_l - s_k} p_i p_j
    """
    n = len(s)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    
    # Ensure non-negative support (clipping for safety during optimization)
    s = np.maximum(s, 0.0)
    p = np.maximum(p, 0.0)
    
    # Re-normalize p if clipping changed sum significantly? 
    # Actually, if we pass raw w to optimizer, it's better to normalize inside.
    # But if we clip p, sum might change.
    # Let's assume s, w are valid enough.
    
    # Build pair sums and probs
    # s_i + s_j
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    # Flatten
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    
    # Cumulative sum of probs
    # C[i] = sum_{j<=i} pp_sorted[j]
    C = np.cumsum(pp_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of pairs with sum < threshold
            # searchsorted returns index where threshold would be inserted
            # side='left' means index of first element >= threshold
            # So elements [0, idx) are < threshold
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            
            P += p[k] * p[l] * prob_inner
            
    return P

def objective(params, n):
    """
    Objective function for scipy.optimize.
    params: array of length 2n. First n are support points, last n are weights.
    We return -P to minimize.
    """
    s = params[:n]
    w = params[n:]
    
    # Basic validity checks to avoid NaNs
    if np.any(np.isnan(s)) or np.any(np.isnan(w)):
        return 1e10
        
    # Clip support to be non-negative
    s = np.maximum(s, 0.0)
    # Clip weights to be non-negative
    w = np.maximum(w, 0.0)
    
    # If all weights are 0, return large penalty
    if np.sum(w) < 1e-9:
        return 1e10
        
    val = evaluate_measure(s, w)
    return -val

def optimize_distribution(n, time_limit=30.0):
    """
    Optimize a distribution with n points.
    """
    start_time = time.time()
    
    # Initial guess: Uniform on [0, 1]
    s0 = np.linspace(0, 1, n)
    w0 = np.ones(n) / n
    
    x0 = np.concatenate([s0, w0])
    
    # Try a few random initializations too
    best_x = x0
    best_val = -1.0
    
    for i in range(5):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            x_init = x0
        else:
            # Random init
            s_rand = np.sort(np.random.uniform(0, 1, n))
            w_rand = np.random.dirichlet(np.ones(n))
            x_init = np.concatenate([s_rand, w_rand])
            
        try:
            res = scipy.optimize.minimize(
                objective, 
                x_init, 
                args=(n,), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x
                
        except Exception as e:
            pass
            
    s_opt = np.maximum(best_x[:n], 0.0)
    w_opt = np.maximum(best_x[n:], 0.0)
    
    # Normalize weights for final return
    w_sum = np.sum(w_opt)
    if w_sum > 1e-9:
        w_opt = w_opt / w_sum
    else:
        # Fallback
        w_opt = np.ones(n) / n
        
    return s_opt, w_opt, best_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different n
    # Small n is fast, large n might be better but slower
    n_values = [10, 20, 30, 50, 80]
    
    for n in n_values:
        if time.time() + 10.0 > deadline - safety_margin:
            break
            
        # Allocate time based on n
        # Larger n is slower per eval, so give it less total time or fewer restarts?
        # Actually, Nelder-Mead complexity depends on dim.
        # Let's just set a time limit per n.
        time_limit = min(60.0, deadline - safety_margin - time.time() - 5.0)
        if time_limit < 5.0:
            break
            
        s, w, score = optimize_distribution(n, time_limit=time_limit)
        
        if score > best_score:
            best_score = score
            best_support = s
            best_weights = w
            
    # Fallback
    if best_support is None:
        n = 10
        best_support = np.linspace(0, 1, n)
        best_weights = np.ones(n) / n
        
    return best_support, best_weights

if __name__ == "__main__":
    # Quick test
    s, w = beat_the_average_measure()
    print(f"Score: {-objective(np.concatenate([s,w]), len(s))}")
