"""Combined program for third autocorrelation inequality optimization.

Uses ADMM with epigraph formulation for max(autoconv) constraint,
combined with L-BFGS-B polishing and coordinate descent.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT (O(N log N) instead of O(N^2))."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]

def _autoconv_fft_grad_h(h, grad_conv):
    """Compute gradient of autoconv w.r.t. h given gradient of autoconv w.r.t. conv.
    
    conv[j+k] = sum_i h[i]*h[i+j+k] for valid indices
    d(conv[j+k])/d(h[m]) = h[m+j+k] + h[m] if m in [0,N) and m+j+k in [0,N)
    
    More efficiently: grad_h[m] = 2 * sum_{j} grad_conv[j] * h[j+m] 
    where j+m is valid index and j is valid conv index.
    Using FFT: this is a cross-correlation.
    """
    n_conv = len(grad_conv)
    n_fft2 = 1
    while n_fft2 < n_conv + N - 1:
        n_fft2 *= 2
    GC = np.fft.rfft(grad_conv, n=n_fft2)
    HH = np.fft.rfft(h, n=n_fft2)
    # Cross-correlation: sum_j grad_conv[j] * h[j+m]
    corr = np.fft.irfft(GC * np.conj(HH), n=n_fft2)
    # We need corr[m] for m in [0, N)
    # But corr from irfft of GC*conj(HH) gives sum_j GC[j]*conj(HH[j]) in frequency
    # Actually, let's use direct approach for correctness
    grad_h = np.zeros(N)
    for m in range(N):
        # grad_h[m] = 2 * sum_{j: 0<=j<n_conv, 0<=j+m<N} grad_conv[j] * h[j+m]
        j_max = min(n_conv, N - m)
        if j_max > 0:
            grad_h[m] = 2.0 * np.dot(grad_conv[:j_max], h[m:m+j_max])
    return grad_h

def _objective_hard(h):
    """Hard objective for final evaluation."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _admm_epigraph(h_init, max_iter=5000, rho=50.0, deadline=None):
    """ADMM for: min_h 2*N*t/(sum(h))^2  s.t. conv(h) <= t*ones
    
    Epigraph form: introduce t >= max(conv(h))
    Augmented Lagrangian:
    L = 2*N*t/(sum(h))^2 + (rho/2)*||conv(h) - t*ones + u||^2 - (rho/2)*||u||^2
    
    ADMM updates:
    1. h-update: min_h 2*N*t/(sum(h))^2 + (rho/2)*||conv(h) - t*ones + u||^2
    2. t-update: min_t 2*N*t/(sum(h))^2 + (rho/2)*||conv(h) - t*ones + u||^2
    3. u-update: u += conv(h) - t*ones
    """
    h = h_init.copy().astype(np.float64)
    conv = _autoconv_fft(h)
    t = max(np.max(conv), 0.0)
    u = np.zeros_like(conv)
    
    s = np.sum(h)
    best_h = h.copy()
    best_val = _objective_hard(h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        conv = _autoconv_fft(h)
        
        # t-update: min_t 2*N*t/s^2 + (rho/2)*||conv - t*ones + u||^2
        # d/dt: 2*N/s^2 + rho * sum(conv - t*ones + u) = 0
        # 2*N/s^2 + rho*(sum(conv) - n_conv*t + sum(u)) = 0
        # t = (sum(conv) + sum(u) + 2*N/(rho*s^2)) / n_conv
        n_conv = len(conv)
        s = np.sum(h)
        if abs(s) < 1e-10:
            s = 1e-10
        t_new = (np.sum(conv) + np.sum(u) + 2.0*N/(rho*s*s)) / n_conv
        t = max(t_new, 0.0)
        
        # h-update: min_h 2*N*t/s^2 + (rho/2)*||conv(h) - t*ones + u||^2
        # This is quadratic in h since conv(h) is quadratic in h
        # Use gradient descent with line search
        # grad_h = 2*rho * d(conv)/d(h) * (conv - t*ones + u)
        # plus gradient of 2*N*t/s^2 which is -4*N*t/s^3 * ones
        
        for inner_it in range(3):
            conv = _autoconv_fft(h)
            residual = conv - t * np.ones(n_conv) + u
            grad_conv = 2.0 * rho * residual
            grad_h = _autoconv_fft_grad_h(h, grad_conv)
            # Add gradient of 2*N*t/s^2 term
            grad_h -= 4.0 * N * t / (s * s * s)
            
            # Gradient step
            lr = 1.0 / (rho * 2.0 * N + 1e-6)
            h_new = h - lr * grad_h
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-10:
                h_new[0] += 1.0
                s_new = np.sum(h_new)
            h = h_new
            s = s_new
        
        # u-update
        conv = _autoconv_fft(h)
        u += conv - t * np.ones(n_conv)
        
        val = _objective_hard(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Adaptive rho
        if it % 100 == 0 and it > 0:
            rho *= 1.2
    
    return best_h

def _soft_max_obj(h, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)

def _soft_max_obj_with_sum_constraint(h, beta=50.0, target_sum=1.0, penalty=100.0):
    """Soft-max objective with soft sum constraint."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        base = 2.0 * N * soft_max / (s * s)
    else:
        base = 2.0 * N * m / (s * s)
    # Penalize deviation from target sum
    sum_penalty = penalty * (s - target_sum) ** 2
    return base + sum_penalty

def _soft_max_obj_grad(h, beta=50.0):
    """Objective and gradient for soft-max objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        soft_max = m + np.log(sum_exp) / beta
        
        # Gradient of soft_max w.r.t. conv
        grad_conv = exp_shifted / sum_exp
        
        # Gradient of conv w.r.t. h
        grad_h = _autoconv_fft_grad_h(h, grad_conv)
        
        val = 2.0 * N * soft_max / (s * s)
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * soft_max / (s * s * s)
        return val, grad_val
    else:
        val = 2.0 * N * m / (s * s)
        idx_max = np.argmax(conv)
        grad_h = np.zeros(N)
        for j in range(N):
            d = idx_max - j
            if 0 <= d < N:
                grad_h[j] = 2.0 * h[d]
        grad_val = 2.0 * N / (s * s) * grad_h - 4.0 * N * m / (s * s * s)
        return val, grad_val

def _energy_flatness_obj(h, beta=50.0, lam=0.05):
    """Objective: soft-max peak + penalty on variance of autoconv entries.
    
    Encourages flat autocorrelation profile (all entries similar),
    which minimizes the peak relative to average.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        base = 2.0 * N * soft_max / (s * s)
    else:
        base = 2.0 * N * m / (s * s)
    
    # Penalize variance of conv entries (encourage flatness)
    conv_mean = np.mean(conv)
    conv_var = np.var(conv)
    # Normalize by mean squared to make scale-invariant
    if abs(conv_mean) > 1e-10:
        flatness_penalty = lam * conv_var / (conv_mean * conv_mean)
    else:
        flatness_penalty = lam * conv_var
    
    return base + flatness_penalty

def _random_walk_init(rng, N, drift=-0.01, noise_scale=0.5):
    """Generate random walk with negative drift as initialization."""
    h = np.zeros(N)
    for i in range(N):
        if i > 0:
            h[i] = h[i-1] + drift + noise_scale * rng.standard_normal()
        else:
            h[i] = noise_scale * rng.standard_normal()
    return h

def _generate_diverse_inits(rng, N):
    """Generate diverse initializations for multi-start optimization."""
    inits = []
    
    # 1. Random normal
    for _ in range(4):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        inits.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        inits.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2 * t)
        h0 += 0.1 * rng.standard_normal(N)
        inits.append(h0)
    
    # 4. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b * block_size:(b + 1) * block_size] = rng.standard_normal()
        inits.append(h0)
    
    # 5. DCT-shaped initializations
    from scipy.fftpack import idct
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        inits.append(h0)
    
    # 6. Random walk initializations
    for _ in range(4):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        inits.append(h0)
    
    return inits

def _coordinate_descent_polish(h, max_iter=200, deadline=None):
    """Polish using coordinate descent on the hard objective with adaptive step."""
    h = h.copy()
    best_val = _objective_hard(h)
    step_size = 0.1
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for j in range(N):
            # Try adaptive perturbations
            for delta in [step_size, -step_size, step_size*0.1, -step_size*0.1]:
                h_test = h.copy()
                h_test[j] += delta
                val = _objective_hard(h_test)
                if val < best_val - 1e-12:
                    h = h_test
                    best_val = val
                    improved = True
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    return h

def _admm_optimize(h, iterations=2000, deadline=None):
    """ADMM-style optimization for the soft-max objective with sum constraint."""
    h = h.copy().astype(np.float64)
    sum_target = np.sum(h)
    
    k_smooth = 30.0
    lr = 0.01
    rho = 10.0  # ADMM penalty
    
    # Initialize dual variable for sum constraint
    nu = 0.0
    
    best_h = h.copy()
    best_val = _objective_hard(h)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        conv = _autoconv_fft(h)
        
        # Smooth max and its gradient
        m = np.max(conv)
        if m > 0:
            shifted = k_smooth * (conv - m)
            shifted = np.clip(shifted, -500, 500)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            sm = m + np.log(sum_exp) / k_smooth
            grad_sm = exp_shifted / sum_exp
        else:
            sm = m
            grad_sm = np.zeros_like(conv)
            idx_max = np.argmax(conv)
            grad_sm[idx_max] = 1.0
        
        # Gradient of smooth max w.r.t. h
        grad_h = _autoconv_fft_grad_h(h, grad_sm)
        
        # Objective gradient: 2*N*sm/(s^2) - 4*N*sm/s^3
        s = np.sum(h)
        if abs(s) < 1e-10:
            s = 1e-10
        obj_grad = 2.0 * N / (s * s) * grad_h - 4.0 * N * sm / (s * s * s)
        
        # Gradient step with sum constraint enforcement
        h_new = h - lr * obj_grad
        
        # Enforce sum constraint
        sum_diff = np.sum(h_new) - sum_target
        h_new -= sum_diff / N
        
        # Update dual variable
        nu += rho * (np.sum(h_new) - sum_target)
        
        h = h_new
        
        val = _objective_hard(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = _generate_diverse_inits(rng, N)
    
    # Optimize each initialization
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        h_curr = h0.copy()
        
        # Phase 1: L-BFGS-B with soft-max objective
        if time.time() < deadline - 25.0:
            try:
                res = minimize(_soft_max_obj, h_curr,
                              method='L-BFGS-B',
                              args=(50.0,),
                              options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < _objective_hard(h_curr):
                    h_curr = res.x
            except Exception:
                pass
        
        # Phase 2: ADMM epigraph optimization
        if time.time() < deadline - 20.0:
            try:
                h_admm = _admm_epigraph(h_curr, max_iter=2000, rho=50.0, deadline=deadline - 15.0)
                val_admm = _objective_hard(h_admm)
                if val_admm < _objective_hard(h_curr):
                    h_curr = h_admm
            except Exception:
                pass
        
        # Phase 3: Energy flatness optimization
        if time.time() < deadline - 15.0:
            try:
                res = minimize(_energy_flatness_obj, h_curr,
                              method='L-BFGS-B',
                              args=(50.0, 0.05),
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < _objective_hard(h_curr):
                    h_curr = res.x
            except Exception:
                pass
        
        # Phase 4: Higher precision soft-max
        if time.time() < deadline - 10.0:
            for beta in [100.0, 200.0]:
                try:
                    res = minimize(_soft_max_obj, h_curr,
                                  method='L-BFGS-B',
                                  args=(beta,),
                                  options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                    val = _objective_hard(res.x)
                    if val < _objective_hard(h_curr):
                        h_curr = res.x
                except Exception:
                    pass
        
        # Update best
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 15.0:
        for beta in [500.0, 1000.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5.0:
        try:
            h_cd = _coordinate_descent_polish(best_h, max_iter=300, deadline=deadline - 2.0)
            val_cd = _objective_hard(h_cd)
            if val_cd < best_val:
                best_val = val_cd
                best_h = h_cd
        except Exception:
            pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
