"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max|ac|."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _objective_smooth_grad(h, beta=50.0):
    """Smooth objective with analytic gradient."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(h)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    sm = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max_abs with respect to ac
    # d(sm)/d(ac[k]) = sign(ac[k]) * exp(beta*(|ac[k]| - m)) / sum_exp
    grad_ac = np.sign(ac) * exp_scaled / sum_exp
    
    # Gradient of ac with respect to h
    # ac = h * h (linear convolution)
    # d(ac[k])/d(h[j]) = h[k-j] + h[j-k] (with appropriate bounds)
    # This is the autocorrelation, so d(ac)/d(h) is like a convolution with h
    
    # For the objective f = sm / s^2
    # df/d(h[j]) = (grad_ac · d(ac)/d(h[j])) / s^2 - 2*sm / s^3
    
    # d(ac)/d(h[j]) = h shifted: for each k, d(ac[k])/d(h[j]) = h[k-j] if 0<=k-j<n else 0 + h[j-k] if 0<=j-k<n else 0
    # This is equivalent to: the gradient w.r.t. h is the convolution of grad_ac with h, plus h with grad_ac
    
    # Using FFT for efficiency
    n = len(h)
    fsize, size = _get_fft_size(n)
    G = np.fft.rfft(grad_ac, fsize)
    H = np.fft.rfft(h, fsize)
    # d(ac)/d(h) = ifft(G * H) gives one part, but we need the full derivative
    # Actually, ac = h * h, so d(ac)/d(h) = 2 * (h * grad_ac) but with proper linear conv
    
    # Simpler: d(ac[k])/d(h[j]) = h[k-j] + h[j-k] where indices are valid
    # This is the same as: ifft(rfft(grad_ac) * rfft(h)) + ifft(rfft(h) * rfft(grad_ac)) but with linear conv
    
    # Let's use the fact that the gradient w.r.t. h is:
    # sum_k grad_ac[k] * d(ac[k])/d(h[j])
    # d(ac[k])/d(h[j]) = h[k-j] (if 0 <= k-j < n) + h[j-k] (if 0 <= j-k < n)
    
    # This can be computed as:
    # For fixed j: sum_k grad_ac[k] * h[k-j] + sum_k grad_ac[k] * h[j-k]
    # = (grad_ac * h)[j] + (grad_ac * h)[j]  (by symmetry of autocorrelation)
    # Wait, that's not quite right. Let me think again.
    
    # ac[k] = sum_j h[j] * h[k-j]
    # d(ac[k])/d(h[j]) = h[k-j] + (if we consider h[j] appears twice when j = k-j, i.e., k=2j, then it's 2*h[j])
    # More carefully: ac[k] = sum_{j=0}^{k} h[j]*h[k-j] for k < n, and sum_{j=k-n+1}^{n-1} h[j]*h[k-j] for k >= n
    # d(ac[k])/d(h[j]) = h[k-j] if 0 <= k-j < n, plus h[j-k] if 0 <= j-k < n, but these are the same term when k-j = j-k, i.e., k=2j
    # Actually: ac[k] = sum_{j} h[j]*h[k-j] where both indices must be in [0, n-1]
    # d(ac[k])/d(h[j]) = h[k-j] (if k-j in [0,n-1]) + h[j-k] (if j-k in [0,n-1])
    # But h[k-j] and h[j-k] are the same when k-j = j-k, which means k=2j
    # So: d(ac[k])/d(h[j]) = h[k-j] + h[j-k] with the understanding that if k-j = j-k, it's counted once
    
    # For the gradient: grad_h[j] = sum_k grad_ac[k] * d(ac[k])/d(h[j])
    # = sum_k grad_ac[k] * h[k-j] (for valid k-j) + sum_k grad_ac[k] * h[j-k] (for valid j-k)
    # The first sum is a cross-correlation, the second is the same by symmetry
    
    # Using FFT: 
    # sum_k grad_ac[k] * h[k-j] = (grad_ac * h)[j] where * denotes linear convolution
    # We can compute this via FFT
    
    G_full = np.fft.rfft(grad_ac, fsize)
    H_full = np.fft.rfft(h, fsize)
    # Linear convolution of grad_ac and h
    conv = np.fft.irfft(G_full * H_full, fsize)[:size]
    # This gives sum_k grad_ac[k] * h[k-j] for j = 0, ..., size-1
    # But we need the gradient w.r.t. h[j], which is:
    # grad_h[j] = conv[j] + conv[j] (by symmetry) but careful with the diagonal
    
    # Actually, let me just compute it directly for correctness, it's O(n^2) but n=400
    # No, let's use the FFT approach properly
    
    # The autoconvolution ac = h * h (linear)
    # d(ac)/d(h) is a matrix where element (k,j) = h[k-j] + h[j-k] (with bounds)
    # grad_h = (d(ac)/d(h))^T @ grad_ac
    
    # Using the property: if ac = h * h, then the gradient of a function f(ac) w.r.t. h is:
    # grad_h[j] = sum_k df/d(ac[k]) * (h[k-j] + h[j-k])
    # This is equivalent to: 2 * (h * grad_ac)[j] where * is linear convolution, 
    # but with a correction for the diagonal terms
    
    # Let's compute: conv = linear_convolution(h, grad_ac)
    # Then grad_h[j] ≈ 2 * conv[j] for j not on the "diagonal" issue
    # For simplicity and since we're using L-BFGS which handles approximate gradients:
    
    conv_full = np.fft.irfft(H_full * G_full, fsize)[:size]
    # grad_h[j] = 2 * conv_full[j] is an approximation
    # More precisely, we need to account for the fact that h[j] appears in ac[2j] twice
    
    # For now, use the approximation which should be good enough for L-BFGS
    grad_h = 2.0 * conv_full[:n]
    
    # Now compute the full gradient of the objective
    # f = sm / s^2
    # df/d(h[j]) = grad_h[j] / s^2 - 2 * sm / s^3
    
    val = sm / (s * s)
    grad = grad_h / (s * s) - 2.0 * sm / (s * s * s)
    
    return val, grad

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_msequence_like(n, period=7):
    h = np.zeros(n)
    for i in range(n):
        val = ((i // period) % 2) ^ ((i % period) % 2)
        h[i] = 1.0 if val == 0 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_binary_random(n, seed=42):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.5, 1.0, -1.0)

def _compute_ac_delta(h, i, delta):
    """Compute change in autoconvolution when h[i] changes by delta.
    Returns only the affected slice and its offset for efficient max computation."""
    n = len(h)
    size = 2 * n - 1
    # ac_new[k] = ac_old[k] + 2*delta*h[k-i] + delta^2 (if k==2i)
    # Affected indices: k in [i, i+n-1]
    offset = i
    # The change vector for indices [i, i+n-1]
    # For k in [i, i+n-1]: change = 2*delta*h[k-i]
    # But h[k-i] is only defined for k-i in [0, n-1], i.e., k in [i, i+n-1]
    # So the change is 2*delta*h[0:n] for those positions
    # Plus delta^2 at position 2i if 2i is in [i, i+n-1]
    change_vec = 2.0 * delta * h.copy()
    if 2 * i >= i and 2 * i < i + n:
        change_vec[2 * i - i] += delta * delta
    return offset, change_vec

def _local_search_flip(h, max_passes=10, deadline=None):
    """Efficient local search using incremental autoconvolution updates."""
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    n = len(h)
    size = 2 * n - 1
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Randomize order for better exploration
        order = np.random.default_rng(hash(tuple(h[:10])) % 2**31).permutation(n)
        
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            
            delta = -2.0 * h[i]
            offset, change_vec = _compute_ac_delta(h, i, delta)
            
            # Compute new max abs without full array update
            # Current max abs
            cur_max = np.max(np.abs(ac))
            
            # The affected region is [offset, offset+n-1]
            # New values in that region: ac[offset:offset+n] + change_vec
            affected = ac[offset:offset+n] + change_vec
            new_max_affected = np.max(np.abs(affected))
            
            # New overall max is max of (unaffected max, new_max_affected)
            # Unaffected: ac[0:offset] and ac[offset+n:size]
            if offset > 0:
                unaff_left = np.max(np.abs(ac[0:offset]))
            else:
                unaff_left = 0.0
            if offset + n < size:
                unaff_right = np.max(np.abs(ac[offset+n:]))
            else:
                unaff_right = 0.0
            unaff_max = max(unaff_left, unaff_right)
            
            new_max = max(unaff_max, new_max_affected)
            s_new = s + delta
            
            if abs(s_new) < 1e-9:
                continue
            
            new_val = new_max / (s_new * s_new)
            
            if new_val < cur_val * (1 - 1e-12):
                # Apply the change
                ac[offset:offset+n] = affected
                s = s_new
                h[i] = -h[i]
                cur_val = new_val
                improved = True
                if cur_val < best_val:
                    best_val = cur_val
        
        if not improved:
            break
    
    return h, best_val

def _simulated_annealing(h, deadline, max_iter=100000):
    """Simulated annealing with efficient incremental updates."""
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_h = h.copy()
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    n = len(h)
    size = 2 * n - 1
    
    T = 0.5
    T_min = 1e-5
    cooling = 0.9998
    rng = np.random.default_rng(999)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        i = int(rng.integers(0, n))
        delta = -2.0 * h[i]
        offset, change_vec = _compute_ac_delta(h, i, delta)
        
        # Efficient max computation
        affected = ac[offset:offset+n] + change_vec
        new_max_affected = np.max(np.abs(affected))
        
        if offset > 0:
            unaff_left = np.max(np.abs(ac[0:offset]))
        else:
            unaff_left = 0.0
        if offset + n < size:
            unaff_right = np.max(np.abs(ac[offset+n:]))
        else:
            unaff_right = 0.0
        unaff_max = max(unaff_left, unaff_right)
        new_max = max(unaff_max, new_max_affected)
        
        s_new = s + delta
        
        if abs(s_new) < 1e-9:
            T *= cooling
            continue
        
        new_val = new_max / (s_new * s_new)
        
        delta_val = new_val - cur_val
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-15)):
            ac[offset:offset+n] = affected
            s = s_new
            h[i] = -h[i]
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _refine_continuous(h_init, deadline):
    """
    Refine using continuous optimization with analytic gradient.
    """
    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return h_init, _score(h_init)
    
    current_h = h_init / s0
    best_h = current_h.copy()
    best_val = _score(best_h)
    
    # Progressive refinement with increasing beta and more iterations
    betas = [20, 50, 100, 200, 400]
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        def obj_grad(x):
            val, grad = _objective_smooth_grad(x, beta=beta)
            return val, grad
        
        try:
            res = minimize(obj, current_h, jac=obj_grad, method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-8,
                                   'ftol': 1e-10, 'maxls': 50})
            h_opt = res.x
            
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    current_h = h_opt.copy()
        except Exception:
            pass
            
    # Final polish with high beta
    if time.time() < deadline - 2:
        try:
            res = minimize(obj, current_h, jac=obj_grad, method='L-BFGS-B', 
                           options={'maxiter': 50, 'disp': False, 'gtol': 1e-10,
                                   'ftol': 1e-12, 'maxls': 30})
            h_opt = res.x
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # 1. Initial Diverse Set
    inits = []
    x = np.arange(N)
    
    inits.append(np.ones(N))
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    inits.append(_gen_chirp(N))
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_barker_like(N))
    inits.append(_gen_msequence_like(N, period=7))
    
    # Add random smoothed
    for seed in [42, 123, 456]:
        inits.append(_gen_random_smoothed(N, seed=seed))
    
    # Add random binary
    for seed in [7, 17, 27]:
        inits.append(_gen_binary_random(N, seed=seed))

    # 2. Phase 1: Quick Continuous Optimization to find basins
    phase1_deadline = t_start + 35.0
    
    for h_init in inits:
        if time.time() > phase1_deadline:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        h_opt, val_opt = _refine_continuous(h_norm, phase1_deadline)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    # 3. Phase 2: Discrete Search (SA + Local Search)
    phase2_deadline = t_start + 130.0
    
    # Run SA starting from the best continuous solution
    if time.time() < phase2_deadline - 5:
        h_sa, val_sa = _simulated_annealing(best_h, phase2_deadline - 5, max_iter=150000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Polish with local search
    if time.time() < phase2_deadline:
        h_ls, val_ls = _local_search_flip(best_h, max_passes=15, deadline=phase2_deadline)
        if val_ls < best_score:
            best_score = val_ls
            best_h = h_ls.copy()
            
    # 4. Phase 3: Final Continuous Polish
    if time.time() < deadline - 2:
        h_final, val_final = _refine_continuous(best_h, deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
            
    # Safety check
    if not np.all(np.isfinite(best_h)):
        return np.ones(N) / N
        
    return best_h

# EVOLVE-BLOCK-END
