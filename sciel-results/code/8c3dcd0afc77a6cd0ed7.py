"""Spectral envelope optimization via LMI relaxation + smooth max surrogate for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import rfft, irfft

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = rfft(x, n=L_fft)
    C = irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_max_and_grad(x, beta=30.0):
    """
    Smooth surrogate: log-sum-exp approximation of max(|ac|) / (sum x)^2.
    Returns (value, gradient).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones(L) / L
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    # Gradient of smooth_max w.r.t. x_j
    # d/dx_j max|ac| ≈ sum_k w_k * sign(ac_k) * d(ac_k)/dx_j
    # ac_k = sum_{m} x_m * x_{k-m} (for valid indices)
    # d(ac_k)/dx_j = sum_{m: j+m=k} x_m + sum_{m: j+m=k} x_m = 2 * sum_{m: m+j=k} x_m (symmetric)
    # Actually ac_k = sum_{m=0}^{N-1} x_m * x_{k-m} where x_i=0 outside [0,N-1]
    # d(ac_k)/dx_j = sum_{m: j+m=k} x_m + sum_{m: j+m=k} x_m ... no
    # ac_k = sum_{m} x_m x_{k-m}. d/dx_j = sum_{m: m=j} x_{k-m} + sum_{m: k-m=j} x_m
    # = x_{k-j} + x_{k-j} = 2*x_{k-j} if both in range, but need care with boundaries.
    # More precisely: d(ac_k)/dx_j = x_{k-j} (if 0<=k-j<N) + x_{k-j} (if 0<=k-j<N) 
    # Wait, that's only if j appears as m OR as k-m. Let me be careful.
    # ac_k = sum_{m=0}^{N-1} x_m * x_{k-m}
    # d(ac_k)/dx_j = x_{k-j} (from the m=j term, if 0<=k-j<N) 
    #                + sum_{m: k-m=j} x_m (from the second factor, if 0<=k-j<N)
    # = x_{k-j} + x_{k-j} = 2*x_{k-j} when 0 <= k-j < N
    # But we also need to handle the case where j appears only in one factor.
    # Actually: ac_k = sum_{m: 0<=m<N, 0<=k-m<N} x_m * x_{k-m}
    # d/dx_j: term where m=j: x_{k-j} (if 0<=k-j<N)
    #          term where k-m=j i.e. m=k-j: x_{k-j} (if 0<=k-j<N)
    # So d(ac_k)/dx_j = 2*x_{k-j} if 0<=k-j<N, else 0 (unless j==k-j, i.e., k=2j, then it's x_j)
    # Wait no: if k=2j, then m=j gives x_j*x_j, derivative is 2*x_j. Still 2*x_j.
    # Hmm, actually when m=j AND k-m=j (i.e., k=2j), it's the same term x_j*x_j, 
    # derivative is 2*x_j. So the formula 2*x_{k-j} works when 0<=k-j<N.
    
    # So grad_smooth_max_j = sum_k w_k * sign(ac_k) * 2 * x_{k-j} (for 0<=k-j<N)
    # Let's define c_k = w_k * sign(ac_k)
    # grad_j = 2 * sum_{k: 0<=k-j<N} c_k * x_{k-j} = 2 * sum_{m=0}^{N-1} c_{j+m} * x_m
    # This is a convolution-like operation.
    
    c = weights * np.sign(ac)
    # c has length 2N-1
    # grad_j = 2 * sum_{m=0}^{N-1} c[j+m] * x[m]
    # This is like a cross-correlation. Let's compute via FFT.
    
    # Define d[m] = x[m] for m in [0,N-1], 0 otherwise
    # Define e[k] = c[k] for k in [0, 2N-2], 0 otherwise
    # We want grad_j = 2 * sum_{m=0}^{N-1} e[j+m] * d[m]
    # = 2 * (e * d)_j where * is cross-correlation... 
    # Actually sum_m e[j+m] d[m] = sum_m e[m+j] d[m]
    # Let f = e reversed: f[k] = e[len(e)-1-k]
    # Then sum_m e[m+j] d[m] = sum_m f[len(e)-1-m-j] d[m] = (f * d)[len(e)-1-j]
    # This is getting complicated. Let me just use a direct approach with FFT.
    
    # Alternative: grad_j = 2 * sum_{m=0}^{N-1} c[j+m] * x[m]
    # Let's pad and use FFT.
    L_total = 2 * N - 1
    L_fft = 1
    while L_fft < 2 * L_total:
        L_fft *= 2
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    c_pad = np.zeros(L_fft)
    c_pad[:L_total] = c
    
    # We want for each j in [0, N-1]: sum_{m=0}^{N-1} c[j+m] * x[m]
    # This is the cross-correlation of c and x evaluated at lag j.
    # corr[j] = sum_m c[m+j] * x[m]
    # In FFT: corr = ifft(fft(c) * conj(fft(x))) but with proper indexing.
    # Actually: corr[j] = sum_m c[m+j] x[m]
    # Let's define: reverse c: cr[k] = c[L_total-1-k]
    # Then corr[j] = sum_m cr[L_total-1-m-j] x[m] = (cr * x)[L_total-1-j]
    # where * is linear convolution.
    
    cr = c[::-1]  # length L_total
    cr_pad = np.zeros(L_fft)
    cr_pad[:L_total] = cr
    
    X_fft = rfft(x_pad, n=L_fft)
    CR_fft = rfft(cr_pad, n=L_fft)
    conv = irfft(X_fft * CR_fft, n=L_fft)
    
    # conv[k] = sum_m cr[m] x[k-m] for linear convolution
    # We want conv[L_total-1-j] for j in [0, N-1]
    # L_total = 2N-1, so L_total-1 = 2N-2
    # conv[2N-2-j] for j=0,...,N-1 gives indices 2N-2, 2N-3, ..., N-1
    
    grad_smooth_max = 2.0 * conv[2*N-2 : 2*N-2-N+1 : -1]
    
    # Now full gradient: d/dx_j [smooth_max / s^2]
    # = (grad_smooth_max_j * s^2 - smooth_max * 2*s) / s^4
    # = grad_smooth_max_j / s^2 - 2*smooth_max / s^3
    
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def _spectral_envelope_init(n_freqs=50):
    """
    Initialize via spectral envelope: choose a PSD shape that minimizes peak autoconv.
    The autoconvolution spectrum is |X(f)|^2. We want to shape |X(f)| to minimize
    the peak of its inverse FFT.
    
    Strategy: parameterize the PSD as a smooth function and optimize its parameters
    to minimize the peak of the corresponding time-domain autoconvolution.
    """
    # Parameterize PSD as exp(-a * f^2) * (1 + b * cos(2*pi*f)) 
    # where f is normalized frequency in [0, 1]
    best_psd = None
    best_peak = np.inf
    
    for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for b in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            f = np.linspace(0, 1, n_freqs)
            psd = np.exp(-a * f**2) * (1.0 + b * np.cos(2 * np.pi * f))
            psd = np.maximum(psd, 0.01)
            
            # Convert to time-domain sequence
            # x is the "amplitude" such that |X|^2 = psd
            # So x = sqrt(psd) in frequency domain, then inverse FFT
            # But we need a real sequence, so let's use the positive frequency part
            freqs = np.linspace(0, 1, n_freqs)
            mag = np.sqrt(psd)
            
            # Create full spectrum (symmetric for real signal)
            full_mag = np.zeros(2 * n_freqs - 1)
            full_mag[:n_freqs] = mag
            full_mag[n_freqs:] = mag[1:n_freqs][::-1]
            
            # Inverse FFT to get time domain
            x_candidate = irfft(full_mag, n=2*n_freqs-1)
            
            # Resample to length N
            if len(x_candidate) >= N:
                x_candidate = x_candidate[:N]
            else:
                x_candidate = np.pad(x_candidate, (0, N - len(x_candidate)))
            
            # Ensure positive (add offset if needed)
            if np.min(x_candidate) < 0:
                x_candidate += abs(np.min(x_candidate)) + 0.1
            
            val = _objective(x_candidate)
            if val < best_peak:
                best_peak = val
                best_psd = x_candidate.copy()
    
    if best_psd is None:
        best_psd = np.ones(N)
    
    return best_psd

def _lmi_relaxation_init():
    """
    LMI-inspired initialization: The peak autoconvolution constraint can be relaxed
    via the Bounded Real Lemma. For a sequence x, the autoconvolution peak is bounded
    by the H-infinity norm of the transfer function. We approximate this by optimizing
    the spectral envelope to be as flat as possible (minimizing dynamic range).
    """
    # Start with a flat spectrum (white noise) and shape it
    rng = np.random.default_rng(123)
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Try various spectral shapes
    for trial in range(20):
        # Random smooth PSD
        freqs = np.linspace(0, 1, 100)
        # Low-pass + ripple
        alpha = rng.uniform(0.5, 5.0)
        beta = rng.uniform(-0.8, 0.8)
        psd = np.exp(-alpha * freqs**2) * (1.0 + beta * np.cos(4 * np.pi * freqs))
        psd = np.maximum(psd, 0.01)
        mag = np.sqrt(psd)
        
        full_mag = np.zeros(199)
        full_mag[:100] = mag
        full_mag[100:] = mag[1:100][::-1]
        
        x_candidate = irfft(full_mag, n=199)
        if len(x_candidate) >= N:
            x_candidate = x_candidate[:N]
        else:
            x_candidate = np.pad(x_candidate, (0, N - len(x_candidate)))
        
        if np.min(x_candidate) < 0:
            x_candidate += abs(np.min(x_candidate)) + 0.1
        
        val = _objective(x_candidate)
        if val < best_val:
            best_val = val
            best_x = x_candidate.copy()
    
    return best_x

def _coordinate_descent(x, max_iters=500, tol=1e-6):
    """
    Coordinate descent on the smooth surrogate.
    For each coordinate j, optimize x_j while keeping others fixed.
    """
    beta = 30.0
    current_val = _smooth_max_and_grad(x, beta)[0]
    x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Optimize x_j: the objective as a function of x_j is smooth
            # Do a few linesearch steps
            x_old = x[j]
            for step in [0.1, -0.1, 0.05, -0.05, 0.01, -0.01]:
                x_trial = x.copy()
                x_trial[j] = x_old + step
                val_trial = _smooth_max_and_grad(x_trial, beta)[0]
                if val_trial < current_val:
                    x[j] = x_trial[j]
                    current_val = val_trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, _objective(x)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    initializations.append(np.ones(N) * np.sign(np.sin(np.pi * t)))
    initializations.append(np.ones(N) * np.sign(np.cos(np.pi * t)))
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * t))
    initializations.append(1.0 - 0.5 * np.sin(2 * np.pi * t))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.sin(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 - 0.3 * np.sin(2 * np.pi * t)))
    
    # Spectral envelope initializations
    try:
        init_spec = _spectral_envelope_init()
        initializations.append(init_spec)
    except Exception:
        pass
    
    try:
        init_lmi = _lmi_relaxation_init()
        initializations.append(init_lmi)
    except Exception:
        pass
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    from scipy.optimize import minimize
    
    betas = [30.0, 50.0, 100.0, 200.0]
    for beta in betas:
        if _check_deadline():
            break
        
        def _obj_grad(x, b=beta):
            return _smooth_max_and_grad(x, beta=b)
        
        try:
            result = minimize(_obj_grad, best_x, method='L-BFGS-B', jac=True,
                             options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Coordinate descent polish
    try:
        x_cd, val_cd = _coordinate_descent(best_x, max_iters=200)
        if val_cd < best_val:
            best_val = val_cd
            best_x = x_cd.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Second L-BFGS pass with higher beta
    for beta in [200.0, 500.0]:
        if _check_deadline():
            break
        
        def _obj_grad2(x, b=beta):
            return _smooth_max_and_grad(x, beta=b)
        
        try:
            result = minimize(_obj_grad2, best_x, method='L-BFGS-B', jac=True,
                             options={'maxiter': 150, 'gtol': 1e-6, 'ftol': 1e-9})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Random perturbation refinement
    rng = np.random.default_rng(42)
    step_size = 0.03
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.998)
        else:
            stall_count += 1
            if stall_count % 3000 == 0:
                step_size = min(0.3, step_size * 1.05)
    
    return best_x
# EVOLVE-BLOCK-END
