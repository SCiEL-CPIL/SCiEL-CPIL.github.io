"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import time

def _compute_ratio(h):
    """Compute the actual ratio for evaluation."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    corr = np.correlate(h, h, mode='full')
    peak = np.max(corr)
    if peak < 1e-30:
        return 0.0
    return l2_sq / (l1 * peak)

def _compute_ratio_and_grad(h):
    """Compute ratio and gradient of the ratio w.r.t. h.
    
    ratio = ||h||_2^2 / (||h||_1 * max(corr))
    where corr = h * h (autoconvolution).
    
    Using subgradient for the max term.
    """
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0, np.zeros_like(h)
    corr = np.correlate(h, h, mode='full')
    peak = np.max(corr)
    if peak < 1e-30:
        return 0.0, np.zeros_like(h)
    
    ratio = l2_sq / (l1 * peak)
    
    # Gradient of ratio = l2_sq / (l1 * peak)
    # d(ratio)/dh_i = [2*h_i * l1 * peak - l2_sq * (h_i/ l1) * peak - l2_sq * l1 * dpeak/dh_i] / (l1 * peak)^2
    # = 2*h_i/(l1*peak) - l2_sq*h_i/(l1^2*peak^2) - l2_sq/(l1^2*peak) * dpeak/dh_i
    
    # dpeak/dh_i: peak = max_k corr[k], corr[k] = sum_j h[j]*h[j+k-N+1] (shifted)
    # For the argmax index k*, d(corr[k*])/dh_i = h[i + k* - (N-1)] + h[i - k*] if in range
    k_star = np.argmax(corr)
    # corr[k] = sum_j h[j] * h[j + k - (N-1)]
    # d(corr[k*])/dh_i: for each j where h[j] appears, the partner is h[j + k* - (N-1)]
    # So d(corr[k*])/dh_i = sum over j: if i == j then h[j+k*-(N-1)], if i == j+k*-(N-1) then h[j]
    # = h[i + k_star - (N-1)] (if valid) + h[i - k_star] (if valid)
    # But we need to be careful: corr is full mode, length 2N-1
    # corr[k] = sum_{j} h[j] * h[j + k - (N-1)] where indices are valid
    
    dpeak_dh = np.zeros(N)
    shift = k_star - (N - 1)
    # corr[k_star] = sum_j h[j] * h[j + shift]
    # d/dh_i corr[k_star] = h[i + shift] (if 0 <= i+shift < N) + h[i - shift] (if 0 <= i-shift < N)
    # But when shift=0, it's 2*h[i]
    if shift == 0:
        dpeak_dh[:] = 2.0 * h
    else:
        # First term: h[i + shift] for valid i
        idx1 = np.arange(N)
        j1 = idx1 + shift
        valid1 = (j1 >= 0) & (j1 < N)
        dpeak_dh[valid1] += h[j1[valid1]]
        # Second term: h[i - shift] for valid i
        j2 = idx1 - shift
        valid2 = (j2 >= 0) & (j2 < N)
        dpeak_dh[valid2] += h[j2[valid2]]
    
    # Gradient
    grad = np.zeros(N)
    grad[:] = 2.0 * h / (l1 * peak) - l2_sq * h / (l1**2 * peak**2) - l2_sq / (l1**2 * peak) * dpeak_dh
    grad = np.where(h > 0, grad, 0.0)  # nonnegativity constraint
    
    return ratio, grad

def _projected_gradient_ascent(h0, max_iter=2000, lr_init=0.1, lr_decay=0.999, 
                                time_limit=None, t0=None):
    """Projected gradient ascent on the ratio with nonnegativity constraint."""
    h = h0.copy()
    h = np.clip(h, 0, None)
    best_h = h.copy()
    best_val = _compute_ratio(h)
    lr = lr_init
    
    for it in range(max_iter):
        if time_limit is not None and t0 is not None:
            if time.time() - t0 > time_limit:
                break
        
        val, grad = _compute_ratio_and_grad(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        # Projected gradient step
        h_new = h + lr * grad
        h_new = np.clip(h_new, 0, None)
        
        # Check if improved
        val_new = _compute_ratio(h_new)
        if val_new > val:
            h = h_new
        else:
            # Try smaller step
            lr *= 0.5
            h_new = h + lr * grad
            h_new = np.clip(h_new, 0, None)
            val_new = _compute_ratio(h_new)
            if val_new > val:
                h = h_new
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        # Gradually increase learning rate back
        lr = min(lr * 1.02, lr_init)
    
    return best_h

def _coordinate_descent(h0, max_iter=5000, time_limit=None, t0=None):
    """Coordinate descent: optimize one coordinate at a time."""
    h = h0.copy()
    h = np.clip(h, 0, None)
    best_h = h.copy()
    best_val = _compute_ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and t0 is not None:
            if time.time() - t0 > time_limit:
                break
        
        improved = False
        for i in range(N):
            if time_limit is not None and t0 is not None:
                if time.time() - t0 > time_limit:
                    break
            
            # For coordinate i, ratio as function of h[i] = x:
            # l2_sq = S2 + x^2 where S2 = sum_{j!=i} h[j]^2
            # l1 = S1 + x where S1 = sum_{j!=i} h[j]
            # peak = max(corr) which depends on x
            # This is complex, so use numerical optimization for single coord
            
            S2 = np.sum(h**2) - h[i]**2
            S1 = np.sum(h) - h[i]
            
            # Find optimal x >= 0
            # Use golden section or simple grid + refinement
            # The function f(x) = (S2 + x^2) / ((S1 + x) * peak(x))
            # peak(x) = max_k corr_k(x) where corr_k depends linearly or quadratically on x
            
            # For simplicity, use a few candidate values
            h[i] = max(h[i], 0)
            candidates = [0.0, h[i] * 0.5, h[i] * 0.7, h[i], h[i] * 1.3, h[i] * 1.5, h[i] * 2.0, 1.0]
            candidates = [c for c in candidates if c >= 0]
            
            best_x = h[i]
            best_f = _compute_ratio(h)
            
            for x in candidates:
                h[i] = x
                f = _compute_ratio(h)
                if f > best_f:
                    best_f = f
                    best_x = x
            
            # Refine around best_x
            if best_x > 0:
                for scale in [0.8, 0.9, 1.1, 1.2]:
                    x_try = best_x * scale
                    h[i] = x_try
                    f = _compute_ratio(h)
                    if f > best_f:
                        best_f = f
                        best_x = x_try
            
            h[i] = best_x
            if best_x != h0[i]:
                improved = True
        
        val = _compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        if not improved:
            break
    
    return best_h

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # safety margin before 150s kill

    best_val = -np.inf
    best_h = np.ones(N)

    def _update(h):
        nonlocal best_val, best_h
        val = _compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        return val

    # Evaluate seed
    _update(best_h)

    rng = np.random.default_rng(2024)
    x = np.linspace(-1, 1, N)

    # Build diverse initializations
    init_list = []

    # 1. Constant
    init_list.append(np.ones(N))

    # 2. Gaussians with various widths
    for sigma in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
        init_list.append(np.exp(-x**2 / (2*sigma**2)))

    # 3. Bimodal peaks
    for gap in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for width in [0.1, 0.15, 0.2, 0.3, 0.4]:
            h_init = np.exp(-((x-gap)**2) / (2*width**2)) + np.exp(-((x+gap)**2) / (2*width**2))
            init_list.append(h_init)

    # 4. Triangular
    init_list.append(np.clip(1 - np.abs(x), 0, None))

    # 5. Cosine window
    init_list.append(0.5 * (1 + np.cos(np.pi * x)))

    # 6. Lorentzian / Cauchy
    for gamma in [0.1, 0.2, 0.3, 0.5, 0.8]:
        init_list.append(1.0 / (1.0 + (x/gamma)**2))

    # 7. Flat-top / box
    for w in [0.3, 0.5, 0.7]:
        h_box = np.where(np.abs(x) <= w, 1.0, 0.0)
        init_list.append(h_box)

    # 8. Random smooth profiles
    for _ in range(12):
        h_rand = np.abs(rng.normal(1.0, 0.9, N))
        h_rand = gaussian_filter1d(h_rand, sigma=rng.uniform(1.0, 3.0))
        h_rand = np.clip(h_rand, 0, None)
        if np.sum(h_rand) > 1e-10:
            init_list.append(h_rand)

    # 9. Sparse-ish profiles
    for _ in range(8):
        h_sparse = np.zeros(N)
        k = rng.integers(3, 18)
        idx = rng.choice(N, size=k, replace=False)
        h_sparse[idx] = rng.uniform(0.3, 1.5, size=k)
        h_sparse = gaussian_filter1d(h_sparse, sigma=rng.uniform(0.5, 1.5))
        h_sparse = np.clip(h_sparse, 0, None)
        if np.sum(h_sparse) > 1e-10:
            init_list.append(h_sparse)

    # 10. Sine-based shapes
    for freq in [1, 2, 3]:
        for phase in [0, np.pi/2]:
            h_sin = np.abs(np.sin(freq * np.pi * x + phase))
            init_list.append(h_sin)

    # 11. Multi-peak (3 peaks)
    for _ in range(6):
        centers = rng.uniform(-0.8, 0.8, size=3)
        widths = rng.uniform(0.1, 0.4, size=3)
        h_multi = np.zeros(N)
        for c, w in zip(centers, widths):
            h_multi += np.exp(-((x-c)**2) / (2*w**2))
        init_list.append(h_multi)

    # Normalize each initialization to sum = N for consistent scale
    for i in range(len(init_list)):
        h0 = np.asarray(init_list[i], dtype=np.float64)
        h0 = np.clip(h0, 0, None)
        s = np.sum(h0)
        if s < 1e-10:
            continue
        init_list[i] = h0 * N / s

    # Remove duplicates (approximate)
    unique_inits = []
    for h0 in init_list:
        if len(unique_inits) == 0:
            unique_inits.append(h0)
            continue
        is_dup = False
        for hu in unique_inits:
            if np.linalg.norm(h0 - hu) < 1e-6 * N:
                is_dup = True
                break
        if not is_dup:
            unique_inits.append(h0)

    # Phase 1: Projected gradient ascent from each initialization
    for i, h0 in enumerate(unique_inits):
        if time.time() > deadline - 10:
            break
        h_opt = _projected_gradient_ascent(h0, max_iter=500, lr_init=0.05,
                                            time_limit=deadline - time.time() - 10, t0=t0)
        _update(h_opt)

    # Phase 2: Coordinate descent on best
    if time.time() < deadline - 5:
        h_cd = _coordinate_descent(best_h, max_iter=2000,
                                    time_limit=deadline - time.time() - 5, t0=t0)
        _update(h_cd)

    # Phase 3: More gradient ascent from best
    if time.time() < deadline - 3:
        h_opt2 = _projected_gradient_ascent(best_h, max_iter=1000, lr_init=0.02,
                                             time_limit=deadline - time.time() - 3, t0=t0)
        _update(h_opt2)

    # Phase 4: Final coordinate descent polish
    if time.time() < deadline - 1:
        h_cd2 = _coordinate_descent(best_h, max_iter=3000,
                                     time_limit=deadline - time.time() - 1, t0=t0)
        _update(h_cd2)

    return list(best_h)
# EVOLVE-BLOCK-END
