"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import fftconvolve

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _softmax_objective(x, beta=50.0):
    """Smooth surrogate: soft-max of |autoconv| / (sum x)^2 using log-sum-exp."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    # soft_max = max_val * exp(beta * (logsumexp(abs_ac/max_val) - log(N_ac)))
    # Actually: soft_max = (1/beta) * log(sum(exp(beta * abs_ac)))
    # For stability: let m = max(abs_ac), then soft_max = m + (1/beta)*log(sum(exp(beta*(abs_ac - m))))
    shifted = beta * (abs_ac - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = max_val + (1.0 / beta) * log_sum
    val = soft_max / (s * s)
    
    # Gradient via chain rule
    # d(soft_max)/d(ac_i) = exp(beta * (abs_ac_i - max_val)) / sum(exp(beta*(abs_ac - max_val)))
    # weighted by sign(ac_i) for the abs
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    d_softmax_d_ac = weights * np.sign(ac)
    
    # d(ac)/dx: ac is autoconvolution, ac[n] = sum_j x[j]*x[j+n]
    # This is complex; use finite difference or analytic
    # For now, return just the value; gradient will be approximated
    return val, d_softmax_d_ac

def _objective_grad(x):
    """Compute objective and gradient using finite differences on the smooth surrogate."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    # Use soft-max with moderate beta for gradient
    beta = 30.0
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation:
    # val = soft_max / s^2
    # d(val)/dx = (1/s^2) * d(soft_max)/dx - (2*soft_max/s^3) * ds/dx
    # ds/dx = 1 (all ones)
    # d(soft_max)/dx = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with appropriate bounds)
    # This is the autocorrelation gradient
    
    # Efficient gradient: d(ac[n])/dx[j] for autoconvolution
    # ac[n] = sum_j x[j]*x[j+n]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (where indices are valid)
    
    # Using FFT: if X = FFT(x), then AC = IFFT(|X|^2)
    # d(AC)/dx can be computed via: d(AC[n])/dx[j] = x[j+n] + x[j-n]
    # In frequency: d(AC)/dX = 2*conj(X) (element-wise for real)
    # So d(AC)/dx = IFFT(2*conj(X) * d(AC)/dX) ... this gets complex
    
    # Simpler: use the fact that gradient of sum_n w[n]*|ac[n]| w.r.t. x
    # = 2 * (w * sign(ac)) convolved with x (with proper handling)
    
    # Let g[n] = weights[n] * sign(ac[n]) for n in [0, 2N-2]
    # Then d(sum_n g[n]*ac[n])/dx[j] = sum_n g[n]*(x[j+n] + x[j-n])
    # = (g * x)[j] + (g_rev * x)[j] where g is zero-padded appropriately
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # d(sum_n g[n]*ac[n])/dx[j] = sum_n g[n]*x[j+n] + sum_n g[n]*x[j-n]
    # First term: (g convolve x)[j] where g is treated as kernel
    # Second term: (g flipped convolve x)[j]
    
    # Use FFT for efficiency
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # First term: sum_n g[n] * x[j+n] = correlation of g and x
    # = IFFT(conj(G) * Xf)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # Second term: sum_n g[n] * x[j-n] = IFFT(G * Xf)
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    ds_dx = np.ones(N)
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * ds_dx
    
    return val, grad

def _objective_from_spectrum(mag):
    """
    Given a power spectrum magnitude profile, compute the objective.
    The autoconvolution's peak relates to the time-domain autocorrelation.
    For a signal with DFT magnitude |X[k]|, the autocorrelation peak is 
    sum |X[k]|^2 (Parseval), and sidelobes relate to off-peak values.
    
    Actually, the autoconvolution in time domain corresponds to:
    ac[n] = sum_k |X[k]|^2 * exp(2*pi*i*k*n/L) / L  (for circular)
    For linear, it's more complex, but we can use the fact that:
    max|ac| / (sum x)^2 = max|ac| / |X[0]|^2 (since sum x = X[0] for the DFT)
    
    So the objective = max_n |ac[n]| / |X[0]|^2
    """
    # mag is the power spectrum: mag[k] = |X[k]|^2 for k=0..N-1
    # X[0] = sum(x), so |X[0]|^2 = mag[0]
    # The autocorrelation in frequency: ac is ifft of |X|^2
    # For the objective, we need max|ac| / mag[0]
    
    if mag[0] < 1e-20:
        return 1e10
    
    # The linear autoconvolution is not simply ifft(|X|^2) due to truncation,
    # but for a rough model, the peak sidelobe ratio is approximately:
    # max(|ac|) / |ac[0]| where ac[0] = sum x^2 for circular... 
    # Actually for autoconvolution: ac[0] = sum_j x[j]^2 (not (sum x)^2)
    # Wait: ac[n] = sum_j x[j]*x[j+n] for n>=0
    # ac[0] = sum_j x[j]^2
    # But the denominator is (sum_j x[j])^2 = |X[0]|^2
    # So the objective is max_n |ac[n]| / |X[0]|^2
    
    # The peak of ac is at n=0: ac[0] = sum x[j]^2
    # But other n can have larger |ac[n]|
    # For the model, if we assume the spectrum is smooth, the max sidelobe
    # is approximately the max of the inverse DFT of the spectrum divided by the DC term.
    
    # Let's compute: ac_circular[n] = ifft(|X|^2)[n]
    # This gives circular autocorrelation. The linear is different but similar.
    # The objective in terms of spectrum: max_n |ifft(mag)[n]| / mag[0]
    
    L_fft = 1
    while L_fft < 2*N-1:
        L_fft *= 2
    
    # For circular autoconvolution of length N, use ifft of mag (padded to N)
    # Actually, the autoconvolution ac[n] = sum_j x[j] x[j+n] for n in [-(N-1), N-1]
    # This is the autocorrelation. In frequency: AC[k] = |X[k]|^2 (for circular)
    # So ac[n] = ifft(|X|^2)[n] for circular case.
    # For linear, it's the same but with zero-padding.
    
    # The peak at n=0: ac[0] = sum x[j]^2 = (1/N) sum_k |X[k]|^2 by Parseval
    # The DC of the spectrum: |X[0]|^2 = (sum x)^2
    # So objective = max_n |ac[n]| / |X[0]|^2
    
    # Let's compute ac via ifft of mag (padded to N for circular)
    ac = np.fft.ifft(mag).real
    max_ac = np.max(np.abs(ac))
    val = max_ac / mag[0]
    return val

def _generate_from_params(params):
    """Generate a step function from spectral parameters."""
    # params[0]: shape parameter for sinc-like spectrum
    # params[1]: tilt parameter
    # params[2]: additional modulation
    
    # A common approach: design |X[k]| to be concentrated at low frequencies
    # with a smooth rolloff, which gives a time-domain signal with low sidelobes.
    
    a = np.clip(params[0], 0.5, 3.0)
    tilt = np.clip(params[1], -1.0, 1.0)
    mod = np.clip(params[2], 0.0, 1.0)
    
    # Frequency bins for DFT of length N
    k = np.arange(N)
    
    # Base: sinc-like spectrum (triangular in frequency -> squared sinc in time)
    # |X[k]| = |sin(pi*a*k/N) / (pi*k/N)| for k != 0, and a for k=0
    # This is the spectrum of a rectangular window of width a*N in time... 
    # Actually, let's use a different parameterization.
    
    # Use a Gaussian-like spectrum: |X[k]| = exp(-c * (k/N)^2)
    # This gives a Gaussian in time, which has low sidelobes.
    c = 0.5 + a  # c in [1, 5.5]
    
    # Frequency grid normalized to [0, 1)
    kf = k / N
    # For real signals, spectrum is symmetric: X[N-k] = conj(X[k])
    # For even N, k=0 and k=N/2 are special.
    
    mag = np.zeros(N)
    # Two-sided: for k=0 to N-1
    # Use a smooth function
    mag = np.exp(-c * kf**2)
    
    # Add tilt: multiply by exp(tilt * kf)
    mag = mag * np.exp(tilt * kf)
    
    # Ensure symmetry for real signal: mag[N-k] should equal mag[k]
    # But we're working with the one-sided DFT... 
    # Actually, for the ifft to give a real signal, we need X[N-k] = conj(X[k])
    # Let's construct the full DFT properly.
    
    # Alternative: just use the magnitude and phase=0, then take ifft and real part.
    # X[k] = mag[k] * exp(i*phase[k]) with phase=0 for simplicity
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))  # normalize
    
    # Add optional modulation
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Try the model-based approach with spectral parameterization
    # Optimize parameters (a, tilt, mod) to minimize the objective
    
    def _param_objective(params):
        x = _generate_from_params(params)
        return _objective(x)
    
    # Initial parameter guesses
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _generate_from_params(result.x)
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Try structured initializations directly
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Gradient-based optimization using soft-max surrogate
    def _obj_func(x):
        val, grad = _objective_grad(x)
        return val, grad
    
    try:
        result = minimize(_obj_func, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Polish with exact objective using BFGS on the non-smooth objective
    try:
        result = minimize(_objective, best_x, method='BFGS', 
                        options={'maxiter': 200, 'gtol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 100000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9998)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.5, step_size * 1.05)
                if stall_count % 25000 == 0:
                    big_pert = rng.normal(0, step_size * 5, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
