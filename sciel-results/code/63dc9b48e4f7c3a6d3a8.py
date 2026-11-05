"""Program for the third autocorrelation inequality: minimize abs(2*N*max(conv)/sum(h)^2)."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_smooth(h, lam=200.0):
    """Smooth surrogate: use log-sum-exp to approximate max(conv)."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    # Normalize for numerical stability
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    shifted = conv / max(conv_max, 1e-12)
    # Log-sum-exp approximation of max
    log_sum_exp = np.log(np.sum(np.exp(lam * (shifted - 1.0)))) / lam + 1.0
    # This approximates max(conv)/conv_max = 1, but scaled
    # Actually let's just use: smooth_max = (1/lam)*log(sum(exp(lam*conv)))
    # But need to be careful with scaling
    # Let's use a simpler approach: smooth_max = (sum(conv^p))^(1/p) for large p
    p = lam
    positive_conv = np.maximum(conv, 0)
    smooth_max = np.sum(positive_conv ** p) ** (1.0 / p)
    raw = 2.0 * n * smooth_max / (s * s)
    return abs(raw)

def _objective_grad(h, lam=100.0):
    """Objective with gradient using smooth max approximation."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    positive_conv = np.maximum(conv, 0)
    
    # Use p-norm with p = lam as smooth max
    p = lam
    # smooth_max = (sum(positive_conv^p))^(1/p)
    # d(smooth_max)/d(conv_i) = (positive_conv_i^(p-1) / sum(positive_conv^p)) * smooth_max
    sp = np.sum(positive_conv ** p)
    if sp < 1e-30:
        smooth_max = 0.0
        grad_conv = np.zeros_like(conv)
    else:
        smooth_max = sp ** (1.0 / p)
        # Gradient of smooth_max w.r.t. conv entries
        grad_conv = np.zeros_like(conv)
        mask = positive_conv > 0
        grad_conv[mask] = positive_conv[mask] ** (p - 1) / sp * smooth_max
    
    raw = 2.0 * n * smooth_max / (s * s)
    val = abs(raw)
    
    # Gradient of smooth_max w.r.t. h
    # smooth_max depends on conv, which depends on h
    # d(conv_m)/d(h_j) = h[m-j] if 0 <= m-j < N, else 0 (for full conv)
    # Actually conv[k] = sum_i h[i]*h[k+i] for appropriate ranges
    # Using FFT: conv = irfft(|FFT(h)|^2)
    # d(conv)/d(h_j) = 2*h (circularly) for the linear part
    
    # Let's compute gradient more carefully
    # conv[m] = sum_{j=0}^{m} h[j]*h[m-j] for m < N
    #         = sum_{j=m-N+1}^{N-1} h[j]*h[m-j] for m >= N
    # d(conv[m])/d(h[j]) = h[m-j] + h[m-j'] where j' is the other index
    # For full convolution: conv[k] = sum_j h[j] h[k-j] where valid indices
    # d(conv[k])/d(h[j]) = h[k-j] (if k-j is valid) + contribution from h[j] being second factor
    
    # Simpler: use the fact that conv = ifft(fft(h)*fft(h))
    # d(conv)/d(h) can be computed via FFT
    
    n_full = 2 * N - 1
    f = np.fft.rfft(h, n_full)
    # conv = irfft(f * f, n_full)
    # d(conv)/d(h_j) = irfft(2 * f * d(f)/d(h_j), n_full)
    # d(f_k)/d(h_j) = exp(-2*pi*i*k*j/n_full)
    # This is getting complex. Let's use a different approach.
    
    # Alternative: use the explicit formula
    # conv[m] = sum_{j: 0<=j<N, 0<=m-j<N} h[j] * h[m-j]
    # d(conv[m])/d(h[j]) = h[m-j] + (h[m-j'] if j != m-j else 0) ... 
    # Actually for each pair (j, m-j), both contribute
    # d(conv[m])/d(h[j]) = h[m-j] if 0 <= m-j < N
    # But also h[j] appears as the second factor when the first index is m-j
    # So d(conv[m])/d(h[j]) = h[m-j] + h[m-j] = 2*h[m-j] if j != m-j
    #                        = h[j] if j == m-j (i.e., m = 2j)
    
    # Let's just compute it directly for the gradient
    dh_conv = np.zeros(N)  # d(smooth_max)/d(h_j)
    
    # For each lag m where conv[m] > 0, add contribution
    for m in range(N):  # Only need lags 0 to N-1 for the gradient w.r.t. h
        if conv[m] > 0 and grad_conv[m] != 0:
            # d(conv[m])/d(h[j]) for j = 0, ..., m
            for j in range(m + 1):
                if j < m - j:
                    dh_conv[j] += 2.0 * h[m - j] * grad_conv[m]
                    dh_conv[m - j] += 2.0 * h[j] * grad_conv[m]
                elif j == m - j:
                    dh_conv[j] += h[j] * grad_conv[m]
    
    # This is O(N^2) which is too slow for N=400 in each iteration
    # Let's use a vectorized approach
    
    # Reset and use vectorized computation
    dh_conv = np.zeros(N)
    # For lag m, the contribution to dh_conv[j] is grad_conv[m] * (2*h[m-j] if j < m-j, etc.)
    # This is equivalent to: dh_conv = some convolution-like operation
    
    # Let's use the FFT-based gradient
    # d(conv)/d(h) where conv = ifft(|f|^2) with f = fft(h, n_full)
    # d(conv_k)/d(h_j) = 2 * Re{conj(f_k) * d(f_k)/d(h_j)} / n_full * ... 
    # This is getting complicated. Let's just use finite differences or a simpler gradient.
    
    # For now, let's use a simpler approach: compute the gradient numerically
    # Or better, let's use the explicit formula with vectorization
    
    # Actually, let's use the property:
    # d(conv)/d(h_j) is the sequence: for each lag m, the derivative is h[m-j] (with appropriate factors)
    # This means d(conv_m)/d(h_j) = h_{m-j} + h_{j-(m-N+1)} ... it's the cross-correlation
    
    # Let's just compute: grad_h = 2*N/(s^3) * (d(conv_max)/dh * s - 2*conv_max * h)
    # where d(conv_max)/dh is computed as above
    
    # For efficiency, let's use a different smooth max: use the top-k average
    # Or just use the actual max and compute its subgradient
    
    # Let's go back to using the actual max and its subgradient
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    if max_val <= 0:
        return 0.0, np.zeros_like(h)
    
    # d(conv[idx])/d(h_j)
    m = idx
    dh_conv_at_max = np.zeros(N)
    for j in range(N):
        km = m - j
        if 0 <= km < N:
            if j != km:
                dh_conv_at_max[j] += 2.0 * h[km]
            else:
                dh_conv_at_max[j] += h[j]
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    # d(val)/d(h_j) = d(raw)/d(h_j) * sign(raw)
    # raw = 2*N*max_val / s^2
    # d(raw)/dh_j = 2*N * (d(max_val)/dh_j * s^2 - max_val * 2*s * h_j) / s^4
    #            = 2*N * (d(max_val)/dh_j * s - 2*max_val*h_j) / s^3
    
    grad = 2.0 * n * (dh_conv_at_max * s - 2.0 * max_val * h) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _objective_and_grad(h):
    """Objective and gradient using actual max."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    if max_val <= 0:
        return 0.0, np.zeros_like(h)
    
    m = idx
    dh_conv_at_max = np.zeros(N)
    for j in range(N):
        km = m - j
        if 0 <= km < N:
            if j != km:
                dh_conv_at_max[j] += 2.0 * h[km]
            else:
                dh_conv_at_max[j] += h[j]
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    grad = 2.0 * n * (dh_conv_at_max * s - 2.0 * max_val * h) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    # 1. All ones (baseline)
    inits.append(np.ones(N))
    
    # 2. Ramp
    inits.append(np.linspace(1, -1, N))
    
    # 3. Cosine bump centered
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    
    # 4. Sharp bump
    inits.append(np.exp(-100 * x**2))
    
    # 5. Wide bump
    inits.append(np.exp(-10 * x**2))
    
    # 6. Two bumps
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    
    # 7. Random with small noise
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    
    # 8. Alternating
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    
    # 9. Triangle wave
    inits.append(1.0 - 2.0 * np.abs(x))
    
    # 10. Sine
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    
    # 11. Three bumps
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    
    # 12. Square wave
    inits.append(np.sign(np.sin(3*np.pi*x)))
    
    # 13. Sawtooth
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    
    # 14. High freq sine
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    
    # 15. Mixed frequency
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    
    # 16. Random with larger variance
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    # 17. Smooth random
    noise = rng.standard_normal(N)
    from scipy.ndimage import gaussian_filter1d
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    
    # 18. Asymmetric ramp
    inits.append(np.linspace(-1, 2, N))
    
    # 19. Double ramp
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    
    # 20. Parabolic
    inits.append(1.0 - 0.5 * x**2)
    
    # 21. Sparse: mostly zero with a few nonzero
    sparse = np.zeros(N)
    sparse[100:150] = 1.0
    inits.append(sparse)
    
    # 22. Very sparse
    sparse2 = np.zeros(N)
    sparse2[150:175] = 2.0
    inits.append(sparse2)
    
    # 23. Block structure
    block = np.zeros(N)
    block[0:100] = 1.0
    block[100:200] = -0.5
    block[200:300] = 1.0
    block[300:400] = -0.5
    inits.append(block)
    
    # 24. Triangular pulse
    tri = np.abs(x)
    inits.append(np.maximum(0, 1 - 2*tri))
    
    # 25. Gaussian with negative lobes
    inits.append(np.exp(-20*x**2) - 0.3*np.exp(-5*x**2))
    
    return inits

def _polish(h, max_iter=300):
    """Do a final polishing optimization from a good starting point."""
    if not _check_time():
        return h
    
    try:
        result = minimize(
            _objective_and_grad,
            h,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 500,
                'maxls': 30,
            }
        )
        return result.x
    except Exception:
        return h

def _refine_with_perturbations(h, rng, n_pert=10, perturb_scale=0.1):
    """Try small perturbations around h and keep the best."""
    best_h = h.copy()
    best_val = _value(h)
    
    for _ in range(n_pert):
        if not _check_time():
            break
        perturbation = perturb_scale * rng.standard_normal(N)
        h_try = best_h + perturbation
        try:
            result = minimize(
                _objective_and_grad,
                h_try,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 300,
                    'maxls': 30,
                }
            )
            val = _value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    return best_h, best_val

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    inits = _make_initial_conditions()
    
    # Phase 1: Try each initial condition with moderate optimization
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 1000,
                    'maxls': 40,
                }
            )
            
            val = _value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish the best result
    if best_val < 2.0 and _check_time():
        polished = _polish(best_h, max_iter=300)
        val = _value(polished)
        if val < best_val:
            best_val = val
            best_h = polished
    
    # Phase 3: Refine with perturbations
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        best_h, best_val = _refine_with_perturbations(best_h, rng, n_pert=15, perturb_scale=0.1)
        
        # Try with smaller scale
        if _check_time():
            best_h, best_val = _refine_with_perturbations(best_h, rng, n_pert=10, perturb_scale=0.03)
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
