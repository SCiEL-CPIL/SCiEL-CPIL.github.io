"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
W = 1.0 / 1200.0  # interval width

# EVOLVE-BLOCK-START

def _compute_ratio(h):
    """Compute max(autoconv) / l2_sq for step function h on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf'), 0.0
    
    # Use FFT on a fine grid for accurate peak detection
    n_grid = 2048
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    peak_idx = int(np.argmax(conv))
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf'), 0.0
    return max_conv / l2_sq, peak_idx

def _peak_and_grad(h, peak_idx):
    """
    Compute the peak value of autoconv and the gradient w.r.t. h at the peak.
    Returns (peak_value, grad_in_h_space, l2_sq).
    The gradient of (autoconv at peak) w.r.t. f is the convolution of f with
    the delta at peak, which in the fine grid is just f shifted to the peak.
    """
    h = np.clip(h, 0.0, None)
    n_grid = 2048
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    # autoconv at peak = sum_k f(x_k) * f(x_peak - x_k)
    # gradient w.r.t. f_j = f(x_peak - x_j)
    # In FFT terms: grad_fine = ifft(F * F) but shifted so that index peak_idx
    # corresponds to the peak. Actually conv[peak_idx] = sum_k f_k * f_{peak - k}
    # d(conv[peak])/d(f_j) = f_{peak - j}
    
    # Compute the full autoconvolution
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    peak_val = conv[peak_idx]
    
    # Gradient in fine grid: grad_fine[j] = f_fine[(peak_idx - j) mod (2*n_grid-1)]
    # But autoconv is defined on [0, 2*n_grid-2], and f is supported on [0, n_grid-1]
    # conv[m] = sum_k f[k] * f[m-k] where f is zero outside [0, n_grid-1]
    # d conv[m] / d f[j] = f[m-j] (with f zero outside support)
    # So grad_fine[j] = f_fine[peak_idx - j] if 0 <= peak_idx - j < n_grid else 0
    
    m = peak_idx
    # For each j in [0, n_grid-1], grad[j] = f[m-j] if 0 <= m-j < n_grid
    j_idx = np.arange(n_grid)
    src = m - j_idx
    valid = (src >= 0) & (src < n_grid)
    grad_fine = np.zeros(n_grid)
    grad_fine[valid] = f_fine[src[valid]]
    grad_fine *= dx  # because conv was scaled by dx
    
    # Project gradient back to h space (600 bins)
    grad_h = np.zeros(N)
    # grad_fine is on the fine grid; map back by averaging over cells
    for i in range(N):
        # cell i covers fine indices where cell_idx == i
        mask = (cell_idx == i)
        if np.any(mask):
            grad_h[i] = np.mean(grad_fine[mask])
    
    l2_sq = np.sum(f_fine ** 2) * dx
    return peak_val, grad_h, l2_sq

def _subspace_line_search(h, subspace_basis, peak_idx, max_iter=50):
    """
    Line search in a low-dimensional subspace spanned by subspace_basis.
    h_new = h + sum_i alpha_i * basis_i
    Minimize peak(h_new) subject to h_new >= 0.
    Uses projected gradient / coordinate ascent in the subspace.
    """
    h = np.clip(h, 0.0, None)
    k = subspace_basis.shape[1]
    alphas = np.zeros(k)
    
    for it in range(max_iter):
        h_cur = h + subspace_basis @ alphas
        h_cur = np.clip(h_cur, 0.0, None)
        
        ratio, pk = _compute_ratio(h_cur)
        if pk != peak_idx:
            # Peak moved; re-evaluate gradient at new peak
            pass
        
        # Compute gradient of peak w.r.t. alphas
        _, grad_h, l2_sq = _peak_and_grad(h_cur, pk)
        if l2_sq < 1e-15:
            break
        grad_alpha = subspace_basis.T @ grad_h
        
        # Also need to account for the l2_sq in denominator
        # Objective: peak / l2_sq. Gradient: (grad_peak * l2_sq - peak * grad_l2) / l2_sq^2
        # grad_l2_sq w.r.t. h = 2 * f (in fine grid), projected
        # For simplicity, do projected gradient on peak only (scale-invariant, so normalize)
        
        # Projected gradient descent on peak
        # But we want to MINIMIZE peak. So move in -grad direction.
        # Use backtracking line search
        lr = 0.1
        improved = False
        for _ in range(20):
            trial_alphas = alphas - lr * grad_alpha
            h_trial = h + subspace_basis @ trial_alphas
            h_trial = np.clip(h_trial, 0.0, None)
            r_trial, _ = _compute_ratio(h_trial)
            if r_trial < ratio:
                alphas = trial_alphas
                improved = True
                break
            lr *= 0.5
        
        if not improved:
            break
    
    h_new = h + subspace_basis @ alphas
    h_new = np.clip(h_new, 0.0, None)
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    i = np.arange(N)
    t = i / N
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform baseline
    candidates.append(np.ones(N))
    
    # 2. Disjoint pulses (every other)
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    candidates.append(h_alt)
    
    # 3. Disjoint pulses (every 3rd)
    h_alt3 = np.zeros(N)
    h_alt3[0::3] = 1.0
    candidates.append(h_alt3)
    
    # 4. Cosine bell
    candidates.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 5. Half cosine (zero at edges)
    candidates.append(np.maximum(0, np.cos(np.pi * (t - 0.5))))
    
    # 6. Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        candidates.append(np.exp(-0.5 * ((t - 0.5) / sigma) ** 2))
    
    # 7. Triangular
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # 8. Power law shapes
    for p in [0.5, 2.0, 4.0]:
        base = 1 - np.abs(2 * t - 1)
        candidates.append(np.maximum(0, base ** p))
    
    # 9. Sinc-like
    candidates.append(np.sinc(2 * t - 1))
    
    # 10. Slight variations around uniform
    for amp in [0.01, 0.02, 0.05, 0.1]:
        candidates.append(1.0 + amp * np.cos(2 * np.pi * t))
    
    # 11. Random smooth candidates
    np.random.seed(42)
    for _ in range(15):
        h_rand = np.random.rand(N)
        kernel = np.ones(7) / 7
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        h_smooth = np.clip(h_smooth, 0, None)
        if np.sum(h_smooth) > 1e-15:
            candidates.append(h_smooth)
    
    # 12. Block patterns (different period)
    for period in [4, 5, 6, 7, 8]:
        h_block = np.zeros(N)
        h_block[0:period//2 : period] = 1.0
        candidates.append(h_block)
    
    # Evaluate all candidates
    best_h = np.ones(N)
    best_ratio = _compute_ratio(best_h)[0]
    best_peak = 0
    
    for h in candidates:
        h = np.clip(h, 0.0, None)
        if np.sum(h) < 1e-15:
            continue
        r, pk = _compute_ratio(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
            best_peak = pk
    
    # Subspace optimization: use random low-dimensional subspaces
    # This is much faster than coordinate descent
    np.random.seed(123)
    h = best_h.copy()
    current_ratio = best_ratio
    current_peak = best_peak
    
    # Number of subspace iterations we can afford
    # Each subspace iteration: ~50 line search steps * 1 FFT each = 50 FFTs
    # Plus 1 FFT for ratio. So ~51 FFTs per subspace iteration.
    # A single FFT is ~0.1ms. So 51 * 0.1ms = 5ms per subspace iteration.
    # In 140s, we can do ~28000 subspace iterations. But let's be conservative.
    
    max_subspace_iters = 2000
    subspace_dim = 3
    
    for it in range(max_subspace_iters):
        if time.time() > deadline - 15.0:
            break
        
        # Generate a random subspace basis
        # Use random smooth vectors to ensure the subspace is "nice"
        basis = np.zeros((N, subspace_dim))
        for j in range(subspace_dim):
            # Random low-frequency signal
            freqs = np.random.randint(1, 20, size=3)
            phases = np.random.uniform(0, 2*np.pi, size=3)
            basis[:, j] = sum(np.sin(2*np.pi*freqs[k]*t + phases[k]) for k in range(3))
            basis[:, j] += np.random.uniform(-0.1, 0.1, size=N)
        
        # Orthonormalize
        Q, _ = np.linalg.qr(basis)
        
        # Line search in this subspace
        h_new = _subspace_line_search(h, Q, current_peak, max_iter=30)
        r_new, pk_new = _compute_ratio(h_new)
        
        if r_new < current_ratio:
            h = h_new
            current_ratio = r_new
            current_peak = pk_new
    
    # Final coordinate descent refinement (fewer iterations since we have a good start)
    h = h.copy()
    step_size = max(0.01, np.max(h) * 0.01)
    
    for iteration in range(200):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        n_trials = 30
        idxs = np.random.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            h[idx] = h_old + step_size
            r_plus, _ = _compute_ratio(h)
            
            h[idx] = h_old - step_size
            r_minus, _ = _compute_ratio(h)
            
            h[idx] = h_old
            
            if r_plus < current_ratio:
                h[idx] = h_old + step_size
                current_ratio = r_plus
                improved = True
            elif r_minus < current_ratio and h_old - step_size > 0:
                h[idx] = h_old - step_size
                current_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    # Final evaluation
    final_r, _ = _compute_ratio(h)
    if final_r < current_ratio:
        current_ratio = final_r
    
    return h.tolist()

# EVOLVE-BLOCK-END
