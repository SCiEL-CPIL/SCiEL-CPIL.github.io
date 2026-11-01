import numpy as np
import time

N = 400

def _autoconv(h):
    """Compute autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _compute_c3(h):
    """Compute C3 = max|autoconv| / (sum|h|)^2"""
    conv = _autoconv(h)
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    if sum_abs == 0:
        return np.inf
    return max_abs / (sum_abs ** 2)

def _tanh_relax(signs, max_iter=2000, tol=1e-10):
    """
    Continuous relaxation: optimize real-valued x where h = tanh(x)
    to minimize max|autoconv| / (sum|tanh(x)|)^2
    """
    x = np.zeros(N)
    # Initialize: positive where signs=1, negative where signs=-1
    x[signs > 0] = 1.0
    x[signs < 0] = -1.0
    
    best_x = x.copy()
    best_val = _compute_c3(np.tanh(x))
    
    for it in range(max_iter):
        h = np.tanh(x)
        conv = _autoconv(h)
        max_idx = np.argmax(np.abs(conv))
        max_val = np.abs(conv[max_idx])
        sum_abs = np.sum(np.abs(h))
        val = max_val / (sum_abs ** 2)
        
        if val < best_val - 1e-15:
            best_val = val
            best_x = x.copy()
        
        # Gradient of max|conv[k]| w.r.t. h
        # conv[k] = sum_i h[i]*h[i+k-N] (for appropriate indexing)
        # Use subgradient: sign of conv at max point
        sgn = np.sign(conv[max_idx])
        
        # Gradient of conv[max_idx] w.r.t. h
        grad_h = np.zeros(N)
        k = max_idx
        # conv[k] = sum_{i} h[i] * h[i + k - N] where indices wrap appropriately
        # For full convolution of length N with itself, length is 2N-1
        # conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
        lo = max(0, k - N + 1)
        hi = min(N - 1, k)
        for i in range(lo, hi + 1):
            j = k - i
            if 0 <= j < N:
                grad_h[i] += h[j]
                if i != j:
                    grad_h[j] += h[i]
                else:
                    grad_h[i] += h[j]  # double count when i==j
        
        # Actually, for autoconvolution: conv[k] = sum_i h[i]*h[i+k-N] 
        # Let me redo this properly
        grad_h = np.zeros(N)
        # conv[k] = sum_{i} h[i] * h[i + k - (N-1)]
        # where the second index must be in [0, N-1]
        offset = k - (N - 1)
        for i in range(N):
            j = i + offset
            if 0 <= j < N:
                grad_h[i] += h[j]
                grad_h[j] += h[i]
        
        # Gradient through tanh: dh/dx = 1 - h^2
        grad_x = sgn * grad_h * (1 - h ** 2)
        
        # Also need gradient of denominator for the ratio
        # f = max_val / sum_abs^2
        # df/dh_i = sgn * grad_h[i] / sum_abs^2 - 2 * max_val * sgn_h[i] / sum_abs^3
        # where sgn_h[i] = sign(h[i]) for |h|
        
        # For simplicity, use L-BFGS-B style with just the numerator gradient
        # scaled appropriately
        lr = 0.1
        x_new = x - lr * grad_x
        
        # Clip to keep tanh in reasonable range
        x_new = np.clip(x_new, -5, 5)
        
        if np.max(np.abs(x_new - x)) < tol:
            break
        x = x_new
    
    return best_x, best_val

def _hill_climb(signs, max_iter=50000):
    """Discrete sign-flip hill climbing to minimize C3."""
    h = signs.copy().astype(float)
    conv = _autoconv(h)
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    best_val = max_abs / (sum_abs ** 2)
    
    for it in range(max_iter):
        improved = False
        # Try flipping each sign
        for i in range(N):
            old_val = h[i]
            h[i] = -h[i]
            
            # Efficient update of autoconvolution
            # Delta in conv from flipping h[i]:
            # conv changes by -2*h[i]_old * h[other] for all pairs involving i
            # This is complex; just recompute for simplicity if N is small enough
            # For N=400, convolve is fast
            conv_new = _autoconv(h)
            max_abs_new = np.max(np.abs(conv_new))
            val_new = max_abs_new / (sum_abs ** 2)
            
            if val_new < best_val - 1e-15:
                best_val = val_new
                conv = conv_new
                max_abs = max_abs_new
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    best_signs = None
    best_score = np.inf
    
    # Generate diverse initial sign patterns
    np.random.seed(42)
    
    initializations = []
    
    # 1. All positive (baseline)
    initializations.append(np.ones(N, dtype=int))
    
    # 2. Alternating
    alt = np.ones(N, dtype=int)
    alt[1::2] = -1
    initializations.append(alt)
    
    # 3. Block patterns
    for block_size in [2, 4, 8, 10, 16, 20, 25, 40, 50, 100]:
        n_blocks = N // block_size
        signs = np.ones(N, dtype=int)
        for b in range(n_blocks):
            if b % 2 == 1:
                signs[b*block_size:(b+1)*block_size] = -1
        initializations.append(signs.copy())
    
    # 4. Random patterns with different densities
    for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
        signs = np.ones(N, dtype=int)
        mask = np.random.rand(N) < density
        signs[mask] = -1
        initializations.append(signs.copy())
    
    # 5. Walsh-like patterns (Hadamard rows)
    # Simple: use bit-reversal patterns
    for period in [3, 5, 7, 11, 13]:
        signs = np.ones(N, dtype=int)
        for i in range(N):
            if (i // period) % 2 == 1:
                signs[i] = -1
        initializations.append(signs.copy())
    
    # Evaluate all initializations
    for signs in initializations:
        if time.time() > deadline - 10:
            break
        h = signs.astype(float)
        score = _compute_c3(h)
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
    
    # Refine top candidates with continuous relaxation + hill climbing
    # Sort initializations by score
    init_scores = []
    for signs in initializations:
        h = signs.astype(float)
        score = _compute_c3(h)
        init_scores.append((score, signs))
    init_scores.sort(key=lambda x: x[0])
    
    # Take top 5 for refinement
    top_k = min(5, len(init_scores))
    
    for rank in range(top_k):
        if time.time() > deadline - 15:
            break
        
        _, signs = init_scores[rank]
        
        # Continuous relaxation
        x_relaxed, val_relaxed = _tanh_relax(signs, max_iter=1000)
        
        # Convert back to signs
        h_relaxed = np.tanh(x_relaxed)
        signs_relaxed = np.sign(h_relaxed)
        signs_relaxed[signs_relaxed == 0] = 1
        
        # Evaluate relaxed version
        h_eval = signs_relaxed.astype(float)
        score_relaxed = _compute_c3(h_eval)
        
        if score_relaxed < best_score:
            best_score = score_relaxed
            best_signs = signs_relaxed.copy()
        
        # Hill climbing from relaxed signs
        if time.time() < deadline - 5:
            h_hc, score_hc = _hill_climb(signs_relaxed, max_iter=20000)
            if score_hc < best_score:
                best_score = score_hc
                best_signs = np.sign(h_hc).astype(int)
                best_signs[best_signs == 0] = 1
        
        # Also hill climb from original
        if time.time() < deadline - 5:
            h_hc2, score_hc2 = _hill_climb(signs, max_iter=20000)
            if score_hc2 < best_score:
                best_score = score_hc2
                best_signs = np.sign(h_hc2).astype(int)
                best_signs[best_signs == 0] = 1
    
    # Final: ensure we return a valid signed step function
    if best_signs is None:
        best_signs = np.ones(N, dtype=int)
    
    # Return as float array (signed step heights)
    return best_signs.astype(float)
