import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    where AutoConv is the continuous autoconvolution of the step function.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _autoconv_peak(h):
    """Compute the autoconvolution peak (max of full convolution)."""
    h_conv = np.convolve(h, h, mode='full')
    return np.max(h_conv)

def _autoconv_peak_index(h):
    """Find the lag index where autoconvolution peaks."""
    h_conv = np.convolve(h, h, mode='full')
    return np.argmax(h_conv)

def _active_set_gradient(h, peak_idx):
    """
    Compute gradient of L2^2 / peak_conv for fixed peak lag.
    For fixed peak at lag k, conv[h,h][k] = sum_i h[i]*h[i+k-N+1] (with appropriate bounds).
    Actually, for full convolution of length 2N-1, index k corresponds to lag (k - (N-1)).
    conv[h,h][k] = sum_{i} h[i] * h[i + k - (N-1)] where valid indices.
    
    Let lag = k - (N-1). Then conv[k] = sum_i h[i]*h[i+lag] for valid i.
    
    d(conv[k])/d(h[j]) = h[j+lag] + h[j-lag] (with boundary handling)
    
    Objective: f = L2^2 / conv_k = (sum h^2) / conv_k
    df/dh_j = 2*h_j / conv_k - (sum h^2) * d(conv_k)/d(h_j) / conv_k^2
    """
    lag = peak_idx - (N - 1)
    l2_sq = np.sum(h**2)
    conv_k = _autoconv_peak(h)
    
    if conv_k == 0:
        return np.zeros(N), conv_k
    
    # Gradient of conv_k w.r.t. h[j]
    grad_conv = np.zeros(N)
    for j in range(N):
        # h[j] appears in conv_k as h[j]*h[j+lag] and h[j+lag]*h[j]
        # First term: h[j] * h[j+lag] if j+lag in [0,N)
        if 0 <= j + lag < N:
            grad_conv[j] += h[j + lag]
        # Second term: h[j-lag] * h[j] if j-lag in [0,N)
        if 0 <= j - lag < N:
            grad_conv[j] += h[j - lag]
    
    # Gradient of f = L2^2 / conv_k
    grad_f = 2.0 * h / conv_k - l2_sq * grad_conv / (conv_k**2)
    
    return grad_f, conv_k

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Diverse parametric initializations ---
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(2, N-2, 20):
        for w in np.linspace(0.5, N/2, 20):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(2, N-2, 20):
        for w in np.linspace(0.5, N, 20):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Edge-concentrated
    for alpha in np.linspace(0.5, 3.0, 15):
        h_edge = np.exp(-alpha * (x_idx - N/2)**2 / N)
        s = _score_raw(h_edge)
        if s > best_score:
            best_score = s
            best_h = h_edge.copy()
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5, 6]:
        for phase in range(period):
            h_alt = np.where((x_idx - phase) % period < period//2, 1.0, 0.0)
            if np.sum(h_alt) > 0:
                s = _score_raw(h_alt)
                if s > best_score:
                    best_score = s
                    best_h = h_alt.copy()

    # 6. Random sparse initialization
    for _ in range(200):
        if time.time() > deadline - 40: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 25)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.uniform(0.3, 1.0, num_peaks)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # 7. Block patterns
    for block_size in [2, 3, 4, 5]:
        for gap in [1, 2, 3]:
            h_block = np.zeros(N)
            for i in range(0, N, block_size + gap):
                h_block[i:i+block_size] = 1.0
            if np.sum(h_block) > 0:
                s = _score_raw(h_block)
                if s > best_score:
                    best_score = s
                    best_h = h_block.copy()

    # --- Phase 2: L-BFGS-B optimization with active-set peak constraint ---
    
    def optimize_active_set(h0, max_iter=500):
        """
        Optimize using L-BFGS-B with active-set identification of dominant convolution lag.
        For a fixed peak lag, the problem becomes smooth and we can use efficient gradient methods.
        """
        h = h0.copy()
        if np.sum(h) == 0:
            return h, 0.0
        
        # Normalize to sum=1
        h = h / np.sum(h)
        
        for outer_iter in range(20):
            if time.time() > deadline - 15:
                break
                
            # Identify current peak lag
            peak_idx = _autoconv_peak_index(h)
            lag = peak_idx - (N - 1)
            
            # Define smooth objective and gradient for fixed peak lag
            def obj_and_grad(h_fixed):
                h_fixed = np.maximum(0, h_fixed)  # Project to nonnegative
                l2_sq = np.sum(h_fixed**2)
                
                # Compute conv at fixed lag
                conv_val = 0.0
                for i in range(N):
                    if 0 <= i + lag < N:
                        conv_val += h_fixed[i] * h_fixed[i + lag]
                
                if conv_val <= 1e-15:
                    return 1e10, np.zeros(N)
                
                # Objective: minimize -L2^2 / conv_val
                f = -l2_sq / conv_val
                
                # Gradient
                grad = np.zeros(N)
                for j in range(N):
                    # d(conv_val)/d(h[j]) = h[j+lag] + h[j-lag] (with bounds)
                    dconv = 0.0
                    if 0 <= j + lag < N:
                        dconv += h_fixed[j + lag]
                    if 0 <= j - lag < N:
                        dconv += h_fixed[j - lag]
                    
                    # df/dh[j] = -2*h[j]/conv_val + l2_sq * dconv / conv_val^2
                    grad[j] = -2.0 * h_fixed[j] / conv_val + l2_sq * dconv / (conv_val**2)
                
                return f, grad
            
            # Use L-BFGS-B
            try:
                res = minimize(obj_and_grad, h, method='L-BFGS-B', 
                              jac=True,
                              bounds=[(0.0, 1.0)] * N,
                              options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.success or res.status in [1, 2, 3]:
                    h_new = res.x
                    h_new = np.maximum(0, h_new)
                    if np.sum(h_new) > 0:
                        h_new = h_new / np.sum(h_new)
                        
                        # Check if peak lag changed
                        new_peak_idx = _autoconv_peak_index(h_new)
                        if new_peak_idx == peak_idx:
                            h = h_new
                        else:
                            # Peak lag changed, continue with new lag
                            h = h_new
            except Exception:
                pass
        
        s_final = _score_raw(h)
        return h, s_final
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Uniform
    starts.append(np.ones(N) / N)
    
    # Gaussians
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(0.5, N/2, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 0:
                h_g /= np.sum(h_g)
                starts.append(h_g)
                
    # Triangles
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(0.5, N, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                starts.append(h_t)
                
    # Sparse patterns
    for _ in range(50):
        h_s = np.zeros(N)
        k = np.random.randint(3, 25)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = np.random.uniform(0.3, 1.0, k)
        if np.sum(h_s) > 0:
            h_s /= np.sum(h_s)
            starts.append(h_s)
            
    # Random smooth
    for _ in range(50):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            h_r /= np.sum(h_r)
            starts.append(h_r)
    
    # Optimize from each start
    for h_start in starts:
        if time.time() > deadline - 20:
            break
        
        if np.sum(h_start) == 0:
            continue
        
        h_opt, s_opt = optimize_active_set(h_start, max_iter=300)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # --- Phase 3: Coordinate descent refinement ---
    
    def coordinate_descent(h, max_iters=100):
        """Iteratively adjust each coordinate to maximize the objective."""
        h = h.copy()
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a grid of values for coordinate i
                candidates = np.unique(np.concatenate([
                    [0.0],
                    np.linspace(0, max(old_val * 2, 0.01), 8),
                    [old_val]
                ]))
                best_cand = old_val
                best_score_i = _score_raw(h)
                
                for cand in candidates:
                    h[i] = cand
                    s_i = _score_raw(h)
                    if s_i > best_score_i:
                        best_score_i = s_i
                        best_cand = cand
                h[i] = best_cand
                if best_cand != old_val:
                    improved = True
            if not improved:
                break
        return h
    
    if time.time() < deadline - 8:
        h_cd = coordinate_descent(best_h, max_iters=80)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd

    # --- Phase 4: Final polish ---
    if time.time() < deadline - 5:
        h_final, s_final = optimize_active_set(best_h, max_iter=500)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    # Final coordinate descent
    if time.time() < deadline - 3:
        h_cd = coordinate_descent(best_h, max_iters=50)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd

    return list(best_h)
