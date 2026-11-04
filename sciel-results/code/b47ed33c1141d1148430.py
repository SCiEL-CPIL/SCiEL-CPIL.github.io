"""
Structural change: Joint spectral magnitude-phase optimization using epigraph formulation.
Instead of fixing magnitudes to be flat and only optimizing phases (current approach),
this approach optimizes both the magnitudes and phases of the FFT coefficients.
The objective is the epigraph form: minimize M subject to |autoconv[k]| <= M for all k.
This is handled by minimizing M + lambda * sum(exp(beta * (|conv[k]| - M))) / beta.
Gradients are computed analytically in the frequency domain and transformed back.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute indices for efficiency
_LAGS = np.arange(2 * N - 1)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_values(x)
    return float(np.max(np.abs(conv)) / (s * s))

def _smooth_metric_epigraph(x, beta=100.0):
    """
    Objective: M + (1/beta) * log(sum(exp(beta * (|conv[k]| - M))))
    This is a smooth approximation of max(|conv|).
    We normalize by sum^2.
    To make it differentiable and stable, we optimize log-M or just M.
    Let's define the objective as:
    f(x) = M / S^2 + (1/beta) * log(sum(exp(beta * (|conv[k]|/S^2 - M/S^2))))
    Actually, it's easier to just minimize the smooth max of |conv|/S^2.
    But the epigraph variable M helps.
    
    Alternative simpler smooth objective for L-BFGS:
    g(x) = (1/S^2) * ( max_val + (1/beta) * log(sum(exp(beta * (|conv| - max_val))) ) )
    This is what the current code does in _smooth_metric.
    
    The KEY DIFFERENCE here is that we allow the SPECTRUM to change.
    In the current code, _try_phase_optimization fixes mags.
    Here, we parameterize the FFT coefficients directly as amplitudes and phases.
    x = ifft(A * exp(i*P)).
    We optimize A (non-negative) and P.
    To ensure x is real, we use the Hermitian symmetry constraints naturally handled by irfft.
    irfft takes a complex array of size FFT_SIZE//2 + 1.
    Let K = FFT_SIZE // 2 + 1.
    fx[0] is real (DC).
    fx[k] for k>0 are complex.
    
    We parameterize:
    fx[0] = a0 (real)
    fx[k] = r[k] * exp(i * theta[k]) for k=1..K-1.
    
    x = irfft(fx, FFT_SIZE)[:N]
    
    We want to minimize metric(x).
    Since metric is scale invariant, we can fix a0 = 1.0 to fix the gauge/scale.
    Then we optimize r[1..K-1] and theta[1..K-1].
    Total variables: (K-1) + (K-1) = 2*(K-1).
    K = 513 (since 2*400-1=799, next power of 2 is 1024, K=513).
    Variables ~ 1024. This is large for L-BFGS but manageable.
    
    However, r[k] must be >= 0. We can parameterize r[k] = exp(log_r[k]) to enforce positivity.
    Or just use bounds [0, inf) in L-BFGS-B.
    
    Let's use L-BFGS-B with bounds.
    """
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    conv = _autoconv_values(x)
    abs_conv = np.abs(conv)
    
    # Smooth max approximation
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    val = soft_max / (s * s)
    
    # Gradient calculation
    # d/dx [ soft_max / s^2 ]
    # = (1/s^2) * d(soft_max)/dx - (2/s^3) * s * d(soft_max)/dx ? No.
    # d(u/v) = (u'v - uv')/v^2
    # u = soft_max, v = s^2
    # u' = d(soft_max)/dx
    # v' = 2s * ds/dx = 2s * 1 (since s = sum(x), ds/dx_j = 1)
    
    # d(soft_max)/dx_j
    # soft_max = max_val + (1/beta) log(sum(exp(beta(|conv_k| - max_val))))
    # d(soft_max)/d(conv_k) = (1/beta) * exp(beta(|conv_k|-max_val)) / sum(...) * beta * sign(conv_k)
    # Let w_k = exp(beta(|conv_k|-max_val)) / sum(exp(...))
    # d(soft_max)/d(conv_k) = w_k * sign(conv_k)
    
    # conv_k = sum_i x_i x_{i-k} (with appropriate bounds)
    # d(conv_k)/dx_j = x_{j-k} + x_{j+k} (if in bounds)
    
    # So d(soft_max)/dx_j = sum_k w_k sign(conv_k) (x_{j-k} + x_{j+k})
    
    # This is a convolution-like operation.
    # Let g_k = w_k * sign(conv_k)
    # d(soft_max)/dx = 2 * (g * x) ? 
    # conv_k = sum_i x_i x_{i-k}.
    # d(conv_k)/dx_j = x_{j-k} + x_{j+k} (circular? No, linear convolution truncated).
    # If we use FFT, it's circular. But our conv is linear.
    # However, if we zero-pad to FFT_SIZE, the circular convolution of the padded signals
    # matches the linear convolution in the first 2N-1 samples.
    # So we can compute the gradient via FFT.
    
    # Let's verify:
    # conv = irfft(fft(x)^2)
    # d(conv)/dx is the gradient of the quadratic form x^T C x? No, conv is a vector.
    # soft_max is a function of the vector conv.
    # d(soft_max)/dx = J_conv^T * d(soft_max)/d(conv)
    # J_conv is the Jacobian of conv wrt x.
    # conv_k = sum_i x_i x_{i-k}
    # This is equivalent to the gradient of the quadratic form?
    # Note that x^T A x = sum_{i,j} A_{ij} x_i x_j.
    # Here we have a specific structure.
    
    # Efficient computation:
    # w = weights for each lag k.
    # grad_soft = 2 * ifft( fft(w) * fft(x) ) ? 
    # Let's check.
    # conv = ifft( Fx^2 ).
    # d(conv_k)/dx_j = ?
    # In frequency domain:
    # F_conv = Fx^2.
    # d F_conv / dx = 2 Fx * d Fx / dx.
    # d Fx / dx is the DFT matrix.
    # This is getting complicated to derive analytically in time domain without loops.
    
    # Given the time budget, let's use the "finite difference check" or a simpler gradient.
    # Actually, the gradient of the log-sum-exp soft max of the autoconvolution
    # can be computed as:
    # grad_x = 2 * ifft( (Fx * conj(Fx)) * something )? No.
    
    # Let's stick to the time-domain gradient computation using the weights w_k.
    # It's O(N^2) per iteration if done naively, which is 160,000 ops.
    # L-BFGS-B might need 100-1000 iterations. 160k * 1000 = 160M ops.
    # NumPy vectorization can help.
    
    # grad_soft_max_j = sum_k w_k sign(conv_k) (x_{j-k} + x_{j+k})
    # This is a convolution of x with h_k = w_k sign(conv_k).
    # h is defined on lags -N+1 to N-1.
    # h[0] = w_0 sign(conv_0)
    # h[k] = w_k sign(conv_k) for k > 0
    # h[-k] = w_k sign(conv_k) for k > 0 (symmetric because conv is symmetric? 
    # conv_k = conv_{-k}. Yes, autoconvolution is even.
    # So h is even.
    # grad_soft_max = 2 * (h * x) ? 
    # Let's check indices.
    # d(conv_k)/dx_j = x_{j-k} + x_{j+k}
    # sum_k w_k sign_k (x_{j-k} + x_{j+k})
    # = sum_k w_k sign_k x_{j-k} + sum_k w_k sign_k x_{j+k}
    # Let m = j-k => k = j-m. Sum_m w_{j-m} sign_{j-m} x_m
    # Let m = j+k => k = m-j. Sum_m w_{m-j} sign_{m-j} x_m
    # Since w and sign are even (w_k = w_{-k}, sign_k = sign_{-k}),
    # w_{j-m} = w_{m-j} and sign_{j-m} = sign_{m-j}.
    # So the two sums are identical.
    # grad_soft_max = 2 * conv(x, h) where h_k = w_k sign(conv_k).
    
    # We can compute this via FFT!
    # h_freq = fft(h, FFT_SIZE)
    # grad_soft_max = 2 * irfft(h_freq * Fx, FFT_SIZE)[:N]
    
    # Construct h
    h = np.zeros(_FFT_SIZE)
    h[:2*N-1] = w * np.sign(conv)
    # h is symmetric, so we just fill the positive lags and copy to negative?
    # No, h is defined on the linear convolution domain.
    # The FFT of h (zero padded) convolved with Fx gives the linear convolution.
    # We need h to be the kernel.
    # h[k] for k=0..N-1 corresponds to positive lags.
    # What about negative lags?
    # The formula sum_k ... x_{j-k} ... covers all integer k.
    # In the FFT convolution, h must be the full kernel.
    # h[k] for k in 0..N-1 is w_k sign_k.
    # h[k] for k in N..2N-2 corresponds to negative lags?
    # In linear convolution of length N and N, the kernel length is 2N-1.
    # Index k in 0..2N-2.
    # k=0: lag 0.
    # k=1: lag 1 (x_{j-1} term? No, standard convolution (h*x)_j = sum_i h_i x_{j-i}).
    # If h_i = w_i sign_i for i>0, and h_{2N-2-i} = w_i sign_i for i>0?
    # Let's just use the property that h is even.
    # h[k] = w_k sign_k for k=0..N-1.
    # h[2N-2-k] = w_k sign_k for k=0..N-1.
    # This creates an even kernel of length 2N-1.
    
    h[0] = w[0] * np.sign(conv[0])
    for k in range(1, N):
        h[k] = w[k] * np.sign(conv[k])
        h[2*N-2-k] = w[k] * np.sign(conv[k])
    
    h_freq = np.fft.rfft(h)
    Fx = np.fft.rfft(x, _FFT_SIZE)
    grad_soft_max = 2.0 * np.fft.irfft(h_freq * Fx, _FFT_SIZE)[:N]
    
    # Now combine with derivative of 1/s^2
    # f = soft_max / s^2
    # df/dx = (1/s^2) grad_soft_max - (2/s^3) s * grad_soft_max ? No.
    # df/dx = (s^2 * grad_soft_max - soft_max * 2s * 1) / s^4
    #        = grad_soft_max / s^2 - 2 * soft_max / s^3
    
    grad = grad_soft_max / (s * s) - 2.0 * soft_max / (s ** 3)
    
    return val, grad

def _spectral_parameterize(params, a0=1.0):
    """
    Convert parameters [log_r, theta] to x.
    log_r: length K-1
    theta: length K-1
    """
    K = _FFT_SIZE // 2 + 1
    log_r = params[:K-1]
    theta = params[K-1:]
    
    r = np.exp(log_r)
    
    fx = np.zeros(K, dtype=complex)
    fx[0] = a0
    fx[1:] = r * np.exp(1j * theta)
    
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    return x

def _spectral_objective(params, beta=100.0):
    x = _spectral_parameterize(params)
    val, _ = _smooth_metric_epigraph(x, beta)
    return val

def _spectral_gradient(params, beta=100.0):
    x = _spectral_parameterize(params)
    val, grad_x = _smooth_metric_epigraph(x, beta)
    
    # Chain rule: dval/dparams = dx/dparams^T * grad_x
    # x = irfft(fx)
    # fx[0] = a0 (const)
    # fx[k] = r[k-1] exp(i theta[k-1])
    
    # dx/d log_r[k-1] = dx/dr[k-1] * r[k-1]
    # dx/d theta[k-1] = dx/dtheta[k-1]
    
    # Let's compute dx/dfx analytically or numerically?
    # irfft is linear.
    # x_j = (1/FFT_SIZE) * ( fx[0] + 2 sum_{k=1}^{K-1} Re(fx[k] exp(-i 2 pi k j / FFT_SIZE)) )
    #     = (1/FFT_SIZE) * ( a0 + 2 sum_{k=1}^{K-1} r[k-1] cos(theta[k-1] - 2 pi k j / FFT_SIZE) )
    
    # dx_j / d log_r[m] (where m=k-1) = (2/FFT_SIZE) * r[m] cos(theta[m] - 2 pi (m+1) j / FFT_SIZE)
    # dx_j / d theta[m] = (2/FFT_SIZE) * r[m] * (-sin(theta[m] - 2 pi (m+1) j / FFT_SIZE))
    
    # This is O(N * K). 400 * 512 = 200k ops.
    # We can vectorize this.
    
    K = _FFT_SIZE // 2 + 1
    log_r = params[:K-1]
    theta = params[K-1:]
    r = np.exp(log_r)
    
    # j = 0..N-1
    j = np.arange(N)
    # k = 1..K-1
    k = np.arange(1, K)
    
    # Phase matrix: phi[j, m] = theta[m] - 2 pi (m+1) j / FFT_SIZE
    # m = 0..K-2, so k = m+1
    # phi = theta[None, :] - 2 * np.pi * k[None, :] * j[:, None] / _FFT_SIZE
    
    # To save memory, we can do this in chunks or just accept the 400x512 matrix (200k floats, 1.6MB).
    phi = theta[None, :] - 2.0 * np.pi * k[None, :] * j[:, None] / _FFT_SIZE
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # dx/dlog_r[j, m] = (2/FFT_SIZE) * r[m] * cos_phi[j, m]
    # dx/dtheta[j, m] = (2/FFT_SIZE) * r[m] * (-sin_phi[j, m])
    
    # grad_params_log_r[m] = sum_j grad_x[j] * (2/FFT_SIZE) * r[m] * cos_phi[j, m]
    # grad_params_theta[m] = sum_j grad_x[j] * (2/FFT_SIZE) * r[m] * (-sin_phi[j, m])
    
    factor = 2.0 / _FFT_SIZE
    r_vec = r * factor
    
    grad_log_r = r_vec * (cos_phi.T @ grad_x)
    grad_theta = -r_vec * (sin_phi.T @ grad_x)
    
    grad_params = np.concatenate([grad_log_r, grad_theta])
    
    return val, grad_params

def _try_spectral_epigraph_optimization(deadline, rng):
    K = _FFT_SIZE // 2 + 1
    n_params = 2 * (K - 1)
    
    # Initialize with random phases and flat magnitudes (log_r=0 => r=1)
    # Or maybe start from the current best if available? 
    # For now, start from a good heuristic: flat spectrum.
    log_r_init = np.zeros(K - 1)
    theta_init = rng.uniform(-np.pi, np.pi, size=K - 1)
    
    params_init = np.concatenate([log_r_init, theta_init])
    
    # Bounds for log_r: [0, 10] (r from 1 to e^10)
    # Bounds for theta: [-pi, pi]
    bounds = [(0, 10)] * (K - 1) + [(-np.pi, np.pi)] * (K - 1)
    
    # L-BFGS-B
    # maxiter should be enough to converge.
    # We check time in the callback? L-BFGS-B doesn't support callback for early stop easily in older scipy, 
    # but we can just run it and hope it finishes. Or use a tight maxiter.
    
    # Let's try multiple restarts.
    best_val = float('inf')
    best_params = params_init.copy()
    
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            p0 = params_init.copy()
        else:
            p0 = np.concatenate([
                rng.uniform(0, 2, size=K-1), # log_r
                rng.uniform(-np.pi, np.pi, size=K-1)
            ])
        
        try:
            res = minimize(
                _spectral_objective,
                p0,
                jac=_spectral_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 1000}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass
    
    # Convert best_params to x
    x_best = _spectral_parameterize(best_params)
    
    # Verify metric
    val_best = _metric(x_best)
    
    return x_best, val_best

def _local_search(x, deadline, rng, max_rounds=50000):
    """Standard local search to polish the result."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Initial guess: ones
    x_init = np.ones(N)
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Phase 1: Spectral Epigraph Optimization
    if time.time() < deadline - 20:
        x_spec, val_spec = _try_spectral_epigraph_optimization(deadline - 20, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
    
    # Phase 2: Local Search Polish
    if time.time() < deadline - 2:
        x_polished, val_polished = _local_search(best_x, deadline - 2, rng)
        if val_polished < best_val:
            best_val = val_polished
            best_x = x_polished
            
    return best_x

# EVOLVE-BLOCK-END
