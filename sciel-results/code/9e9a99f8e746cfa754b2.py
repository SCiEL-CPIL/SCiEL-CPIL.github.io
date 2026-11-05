"""Program using smooth max surrogate in Fourier domain for autocorrelation inequality."""
N = 400
import numpy as np
import time
from scipy.optimize import minimize

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 142.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max_surrogate(conv, beta):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns (smooth_max, gradient_wrt_conv).
    """
    # Subtract max for numerical stability
    m = np.max(conv)
    shifted = conv - m
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(beta * shifted)
    lse = np.log(np.sum(exp_shifted)) / beta + m
    weights = exp_shifted / np.sum(exp_shifted)
    grad = weights
    return lse, grad

def _fourier_to_h(coeffs):
    """Convert Fourier coefficients (real, even symmetry) to time-domain h."""
    n = len(coeffs)
    # coeffs[0] is DC, coeffs[1..n-1] are cosine coefficients
    h = np.zeros(N)
    h[:] = coeffs[0]
    for k in range(1, n):
        h += coeffs[k] * np.cos(2 * np.pi * k * np.arange(N) / N)
    return h

def _h_to_fourier(h):
    """Convert time-domain h to Fourier cosine coefficients."""
    n = N
    coeffs = np.zeros(n)
    coeffs[0] = np.mean(h)
    for k in range(1, n):
        coeffs[k] = (2.0 / n) * np.sum(h * np.cos(2 * np.pi * k * np.arange(n) / n))
    return coeffs

def _objective_fourier(coeffs, beta):
    """
    Objective in Fourier domain with smooth max surrogate.
    Returns (value, gradient).
    """
    h = _fourier_to_h(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    conv = _autoconv(h)
    smooth_max, grad_wrt_conv = _smooth_max_surrogate(conv, beta)
    
    raw = 2.0 * N * smooth_max / (s * s)
    val = abs(raw)
    
    # Gradient with respect to h
    # d(smooth_max)/dh = sum_k grad_k * dh_conv_k/dh
    # conv[m] = sum_j h[j] * h[m-j]
    # d(conv[k])/dh[j] = h[k-j] + h[j-k] (with appropriate bounds)
    
    # For each lag k, d(conv[k])/dh[j] = h[k-j] if 0<=k-j<N else 0, plus h[j-k] if 0<=j-k<N else 0
    # Actually conv[k] = sum_j h[j] * h[k-j] where indices wrap? No, full convolution.
    # conv[k] = sum_{j: 0<=j<N, 0<=k-j<N} h[j] * h[k-j]
    # d(conv[k])/dh[j] = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    # But since we're doing full convolution (not circular), let me be careful.
    
    # Actually for full convolution: conv[k] = sum_j h[j]*h[k-j] where j ranges over valid indices
    # d(conv[k])/dh[j] = h[k-j] if 0 <= k-j < N, else 0
    # Plus: if we consider that h appears twice, d(conv[k])/dh[j] also gets contribution from h[j]*h[k-j] where j is the second index
    # So d(conv[k])/dh[j] = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    
    dh_conv = np.zeros(N)
    for k in range(2 * N - 1):
        gk = grad_wrt_conv[k]
        if abs(gk) < 1e-15:
            continue
        for j in range(N):
            # Contribution from h[j] * h[k-j]
            if 0 <= k - j < N:
                dh_conv[j] += gk * h[k - j]
            # Contribution from h[j] being the second factor: h[j]*h[k-j] -> d/dh[j] of h[j]*h[k-j] = h[k-j]
            # Wait, this is the same term. Let me reconsider.
            # conv[k] = sum_j h[j] * h[k-j]
            # d(conv[k])/dh[j] = h[k-j] (from the first h[j]) + h[j] * 0 (h[k-j] doesn't depend on h[j] unless k-j=j)
            # Actually: conv[k] = sum_j h[j] * h[k-j]
            # d/dh[j] conv[k] = h[k-j] (derivative of h[j] term) + sum_{j'} h[j'] * d/dh[j] h[j'-j] ... no
            # Let me just do: conv[k] = sum_j h[j] * h[k-j]
            # d(conv[k])/dh[j] = h[k-j] + (if there's a j' such that j' - j = k, i.e., j' = k+j, then h[j'] contributes)
            # Wait no. conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where h[k-j]=0 if k-j not in [0,N-1]
            # d(conv[k])/dh[j] = h[k-j] (from differentiating the h[j] factor)
            #           + sum_{j'} h[j'] * d/dh[j] h[j'-j] ... this is zero unless j'-j = j, i.e., j'=2j, but that's not how it works
            # Actually h[k-j] is a fixed index, so d/dh[j] of h[k-j] is 0 unless k-j = j, i.e., k=2j
            # So d(conv[k])/dh[j] = h[k-j] + h[j] * delta_{k-j, j} = h[k-j] + h[j] * delta_{k, 2j}
            # Hmm, this is getting complicated. Let me use a different approach.
            
            # Actually, the cleanest way: conv = h * h (full convolution)
            # d(conv[k])/dh[j] = h[k-j] if 0<=k-j<N, plus we need to account for the fact that h appears in both positions
            # For full convolution: conv[k] = sum_j h[j] h[k-j]
            # d/dh[j] conv[k] = h[k-j] + h[j] * (d/dh[j] h[k-j])
            # But h[k-j] is a specific entry, so d/dh[j] h[k-j] = 1 if k-j=j else 0 = 1 if k=2j else 0
            # So d(conv[k])/dh[j] = h[k-j] + h[j] * (1 if k==2j else 0)
            pass
    
    # Let me redo this more carefully using the structure
    # conv[k] = sum_j h[j] * h[k-j]
    # d(conv[k])/dh[j] = h[k-j] (if 0<=k-j<N) + h[j] * (1 if k-j==j else 0)
    # The second term: k-j==j means k=2j, so this is h[j] if k==2j and 0<=k-j<N (which is 0<=j<N, always true)
    
    # Let me compute gradient numerically for safety, or use a cleaner formula
    # Actually, let me just use finite differences for the gradient in Fourier space as a fallback,
    # but that's expensive. Let me think again.
    
    # Alternative: compute gradient w.r.t. h first, then map to Fourier
    # d(conv[k])/dh[j] = h[k-j] if 0<=k-j<N, else 0, PLUS h[j] if k==2j (from the symmetric term)
    # Wait, I think I'm overcomplicating. Let me use:
    # conv[k] = sum_j h[j] h[k-j]
    # This is a bilinear form. d(conv[k])/dh[j] = h[k-j] + h[j] * d(k-j)/dj... no.
    # h[k-j] is just the value at index k-j. It doesn't depend on j in a differentiable way beyond the index.
    # So d(conv[k])/dh[j] = h[k-j] (treating h[k-j] as constant w.r.t. h[j]) 
    #   PLUS: if k-j = j (i.e., k=2j), then h[k-j] = h[j], so we get an extra h[j] term
    #   Actually no: conv[k] = sum_j h[j] h[k-j]. The term involving h[j] is h[j]*h[k-j].
    #   d/dh[j] of h[j]*h[k-j] = h[k-j] + h[j]*d/dh[j](h[k-j])
    #   d/dh[j](h[k-j]) = 1 if k-j == j, else 0 = 1 if k==2j else 0
    #   So d(conv[k])/dh[j] = h[k-j] (if 0<=k-j<N) + h[j] (if k==2j)
    
    # OK let me just compute it properly
    dh_conv = np.zeros(N)
    for k in range(2 * N - 1):
        gk = grad_wrt_conv[k]
        if abs(gk) < 1e-15:
            continue
        for j in range(N):
            term = 0.0
            if 0 <= k - j < N:
                term += h[k - j]
            if k == 2 * j:
                term += h[j]
            dh_conv[j] += gk * term
    
    # d(val)/dh = d(abs(raw))/dh
    # raw = 2N * smooth_max / s^2
    # d(raw)/dh = 2N * (dh_smooth_max * s^2 - smooth_max * 2s * dh_s) / s^4
    #           = 2N * (dh_smooth_max * s - 2 * smooth_max * dh_s) / s^3
    # where dh_smooth_max = sum_k grad_k * d(conv[k])/dh = grad_wrt_conv @ (convolution Jacobian)
    # and dh_s = sum_j 1 * dh[j]... wait, s = sum h[j], so ds/dh[j] = 1 for all j
    
    # d(smooth_max)/dh[j] = sum_k grad_wrt_conv[k] * d(conv[k])/dh[j] = (dh_conv)[j]
    
    ds_dh = np.ones(N)
    d_raw_dh = 2.0 * N * (dh_conv * s - 2.0 * smooth_max * ds_dh) / (s ** 3)
    
    if raw < 0:
        d_raw_dh = -d_raw_dh
    
    # Now map from h-space to Fourier coefficient space
    # h = F @ coeffs, where F is the Fourier cosine matrix
    # coeffs = F^T @ h (approximately, with proper normalization)
    # d(val)/d(coeffs[k]) = sum_j d(val)/dh[j] * d(h[j])/d(coeffs[k])
    # d(h[j])/d(coeffs[k]) = cos(2*pi*k*j/N) for k>=1, 1 for k=0
    # With our normalization: h[j] = coeffs[0] + sum_{k=1}^{n-1} coeffs[k] * cos(2*pi*k*j/N)
    # So d(h[j])/d(coeffs[k]) = cos(2*pi*k*j/N) for k>=1, 1 for k=0
    
    n_coeffs = len(coeffs)
    grad_coeffs = np.zeros(n_coeffs)
    grad_coeffs[0] = np.sum(d_raw_dh)
    for k in range(1, n_coeffs):
        grad_coeffs[k] = np.sum(d_raw_dh * np.cos(2 * np.pi * k * np.arange(N) / N))
    
    return val, grad_coeffs

def _objective_fourier_simple(coeffs, beta):
    """Simpler objective that just evaluates the value without gradient, for use with numerical grad."""
    h = _fourier_to_h(coeffs)
    return _compute_value(h)

def _make_initial_fourier():
    """Generate initial conditions in Fourier space."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    # Start with all ones
    h0 = np.ones(N)
    inits.append(_h_to_fourier(h0))
    
    # Various shapes
    shapes = [
        np.ones(N),
        np.linspace(1, -1, N),
        1.0 + 0.5 * np.cos(2 * np.pi * x),
        np.exp(-100 * x**2),
        np.exp(-10 * x**2),
        np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2),
        1.0 + 0.1 * rng.standard_normal(N),
        np.where(np.arange(N) % 2 == 0, 1.0, -0.5),
        1.0 - 2.0 * np.abs(x),
        1.0 + 0.5 * np.sin(4 * np.pi * x),
        np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2),
        np.sign(np.sin(3*np.pi*x)),
        2.0 * (x - np.floor(x + 0.5)),
        1.0 + 0.3 * np.sin(8 * np.pi * x),
        1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x),
        1.0 + 0.3 * rng.standard_normal(N),
    ]
    
    for shape in shapes:
        inits.append(_h_to_fourier(shape.astype(np.float64)))
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    n_coeffs = 80  # Use 80 Fourier coefficients (DC + 79 harmonics)
    
    inits = _make_initial_fourier()
    
    # Phase 1: Optimize in Fourier space with smooth max surrogate
    beta_values = [10.0, 20.0, 50.0, 100.0]
    
    for beta in beta_values:
        if not _check_time():
            break
        
        for init_coeffs in inits:
            if not _check_time():
                break
            
            # Truncate to n_coeffs
            c0 = init_coeffs[:n_coeffs].copy()
            
            try:
                result = minimize(
                    _objective_fourier,
                    c0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': 500,
                        'maxls': 30,
                    }
                )
                
                h_try = _fourier_to_h(result.x)
                val = _compute_value(h_try)
                
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
            except Exception:
                continue
    
    # Phase 2: Polish in time domain with exact max
    if best_val < 2.0 and _check_time():
        def _objective_time(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10, np.zeros_like(h)
            conv = _autoconv(h)
            max_val = np.max(conv)
            idx = np.argmax(conv)
            raw = 2.0 * N * max_val / (s * s)
            val = abs(raw)
            
            # Gradient
            dh_conv = np.zeros(N)
            km = idx
            for j in range(N):
                if 0 <= km - j < N:
                    dh_conv[j] += h[km - j]
                if km == 2 * j:
                    dh_conv[j] += h[j]
            
            grad = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
            if raw < 0:
                grad = -grad
            
            return val, grad
        
        try:
            result = minimize(
                _objective_time,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 800,
                    'maxls': 40,
                }
            )
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Random restarts in Fourier space if time remains
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(999)
        best_coeffs = _h_to_fourier(best_h)[:n_coeffs]
        
        for _ in range(3):
            if not _check_time():
                break
            pert = 0.1 * rng.standard_normal(n_coeffs)
            c_try = best_coeffs + pert
            try:
                result = minimize(
                    _objective_fourier,
                    c_try,
                    args=(50.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 150,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': 400,
                        'maxls': 30,
                    }
                )
                h_try = _fourier_to_h(result.x)
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
            except Exception:
                continue
    
    # Final verification
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    val = _compute_value(best_h)
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
