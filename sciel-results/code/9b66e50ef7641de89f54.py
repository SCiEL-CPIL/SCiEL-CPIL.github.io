"""Spectral phase optimization + L-BFGS + coordinate descent + peak-targeted polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _objective_only(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta=50.0):
    """
    Compute soft-max of |ac| and its gradient w.r.t. x using FFT.
    Soft-max: f = (1/beta) * log(sum_j exp(beta * |ac[j]|))
    For large beta, this approximates max_j |ac[j]|.
    
    The autoconvolution ac[n] = sum_k x[k] x[k+n] for n in [-(N-1), N-1].
    In frequency domain: AC = IFFT(|FFT(x)|^2) for circular, but for linear
    we use the padded FFT.
    
    Gradient: d/dx_i [ac[n]] = x[i+n] + x[i-n] (with boundary handling)
    For the soft-max: grad_i = sum_n w_n * (x[i+n] + x[i-n]) / (s^2 * beta)
    where w_n = exp(beta*|ac[n]|) / sum_j exp(beta*|ac[j]|)
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max with temperature
    scaled = beta * abs_ac
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_exp if False else exp_scaled)
    if sum_exp < 1e-300:
        sum_exp = 1e-300
    
    weights = exp_scaled / sum_exp
    soft_max_val = (max_scaled + np.log(sum_exp)) / beta
    
    # Objective: soft_max / s^2
    obj = soft_max_val / (s * s)
    
    # Gradient of soft_max w.r.t. x:
    # d(soft_max)/dx_i = (1/beta) * sum_n weights[n] * d|ac[n]|/dx_i
    # d|ac[n]|/dx_i = sign(ac[n]) * (x[i+n] + x[i-n]) with bounds
    # For linear autoconvolution ac[n] for n in [0, 2N-2]:
    # ac[n] = sum_{k} x[k] * x[k+n-N+1] ... this is getting complex.
    # Let's use the FFT-based gradient.
    
    # Actually, for the linear autoconvolution:
    # ac[n] = sum_{k=0}^{N-1-n} x[k] * x[k+n] for n >= 0
    # ac[n] = sum_{k=n}^{N-1} x[k] * x[k-n] for n < 0 (by symmetry ac[-n] = ac[n])
    
    # The gradient of ac[n] w.r.t. x[i] is:
    # For n >= 0: x[i+n] if i+n < N, plus x[i-n] if i-n >= 0 (from the symmetric part)
    # Wait, ac[n] = sum_k x[k]*x[k+n]. d/dx_i ac[n] = x[i+n] (if i+n < N) + x[i-n] (if i-n >= 0, from ac[-n])
    # But ac is defined for n in [-(N-1), N-1], and ac is even: ac[n] = ac[-n].
    # So |ac[n]| for n in [0, N-1] covers all unique values.
    
    # Let me recompute. The linear autoconvolution has 2N-1 values indexed 0 to 2N-2.
    # ac[m] = sum_k x[k] * x[k+m-N+1] ... no.
    # Standard: ac[m] for m in [0, 2N-2]: ac[m] = sum_{k} x[k] * x[k-(m-N+1)]
    # Actually, let's just use: the autoconvolution of x with itself.
    # ac[m] = sum_{k=0}^{N-1} x[k] * x[k+m] where we define x[j]=0 for j<0 or j>=N.
    # So ac[m] = sum_{k=max(0,-m)}^{N-1-max(0,m)} x[k] * x[k+m]
    
    # d/dx[i] ac[m] = x[i+m] (if 0<=i+m<N) + x[i-m] (if 0<=i-m<N)
    # because x[i] appears in ac[m] as x[i]*x[i+m] and as x[i-m]*x[i]
    
    # For the soft-max over all m in [0, 2N-2]:
    # grad_obj[i] = (1/s^2) * (1/beta) * sum_m weights[m] * sign(ac[m]) * (x[i+m]+x[i-m])
    #              - (2/s^3) * soft_max_val * (x[i] + ...) ... wait, denominator is s^2
    # d/dx_i [f/s^2] = (f' * s^2 - f * 2*s) / s^4 = (f'/s^2) - 2*f*x[i]/s^3
    
    # This is expensive to compute directly. Let's use FFT.
    # The gradient of the circular autoconvolution is easier.
    # For now, let's approximate: use the fact that the gradient of max|ac| w.r.t. x
    # is approximately the "matched filter" response.
    
    # Simpler approach: numerical gradient or just use the weights to guide perturbations.
    # For L-BFGS, we need a proper gradient. Let me compute it via the FFT trick.
    
    # The autoconvolution in frequency: if we zero-pad x to length L, then
    # FFT(ac_padded) = |FFT(x_padded)|^2
    # So ac = IFFT(|X|^2) where X = FFT(x, n=L)
    
    # d/dx_i |X[k]|^2 = 2*Re(conj(X[k]) * dX[k]/dx_i)
    # dX[k]/dx_i = exp(-2*pi*i*k*i/L)
    # So d/dx_i |X[k]|^2 = 2*Re(conj(X[k]) * exp(-2*pi*i*k*i/L))
    
    # And ac = IFFT(|X|^2), so d ac[m]/dx_i = (1/L) * sum_k d|X[k]|^2/dx_i * exp(2*pi*i*k*m/L)
    
    # This is still O(N*L). Let's just do the direct computation for N=400.
    # 2N-1 = 799 values, each requiring O(N) work -> O(N^2) = 160000 per gradient.
    # That's fine for L-BFGS with a few hundred iterations.
    
    # Compute gradient of soft_max w.r.t. x
    # grad_soft[i] = (1/beta) * sum_m weights[m] * sign(ac[m]) * (x[i+m] + x[i-m])_clipped
    # where the sum is over m in [0, 2N-2]
    
    # Let's compute this efficiently.
    # For each i, grad_soft[i] = (1/beta) * sum_m w[m] * sgn[m] * (x[i+m] if i+m<N else 0 + x[i-m] if i-m>=0 else 0)
    
    # We can split: sum_m w[m]*sgn[m]*x[i+m] + sum_m w[m]*sgn[m]*x[i-m]
    # = (w*sgn convolved with x)[i] + (w*sgn convolved with x reversed)[i] ... not quite.
    
    # Let h[m] = w[m] * sign(ac[m]) for m in [0, 2N-2]
    # Then sum_m h[m] * x[i+m] = (x * h)[i] where * is cross-correlation
    # And sum_m h[m] * x[i-m] = (x * h)[i] with h reversed... 
    
    # Actually: sum_m h[m] x[i+m] = sum_k h[k-i] x[k] (letting k=i+m, m=k-i)
    # This is the cross-correlation of x with h at lag i.
    # Similarly sum_m h[m] x[i-m] = sum_k h[k-i] x[k] ... wait.
    # sum_m h[m] x[i-m]: let k = i-m, m = i-k. sum_k h[i-k] x[k].
    # This is the cross-correlation of h with x at lag i.
    
    # So grad_soft[i] = (1/beta) * ( (x * h)[i] + (h * x)[i] ) where * denotes cross-correlation
    # But cross-correlation of x with h at lag i is sum_k x[k] h[k-i]
    # And cross-correlation of h with x at lag i is sum_k h[k] x[k-i]
    
    # These are related: (h * x)[i] = (x * h)[-i]
    # So grad_soft[i] = (1/beta) * ( (x*h)[i] + (x*h)[-i] )
    
    # Where (x*h)[i] = sum_k x[k] h[k-i] for i in [-(2N-2), 2N-2]
    
    # We can compute this via FFT!
    L_fft = 1
    while L_fft < 4*N:
        L_fft *= 2
    
    X_fft = np.fft.fft(x, n=L_fft)
    H_fft = np.fft.fft(h, n=L_fft)
    corr = np.fft.ifft(X_fft * np.conj(H_fft)).real  # cross-correlation x with h
    
    # corr[i] = sum_k x[k] h[k-i] for i in [0, L_fft-1] (circular)
    # We need corr[i] and corr[-i mod L_fft]
    grad_soft = (corr + np.roll(corr, 1)) / beta  # approximate
    
    # Actually this is getting complicated. Let me just do the direct O(N^2) computation.
    grad_soft = np.zeros(N)
    for i in range(N):
        total = 0.0
        for m in range(2*N - 1):
            # x[i+m] if i+m < N
            if i + m < N:
                total += weights[m] * np.sign(ac[m]) * x[i + m]
            # x[i-m] if i-m >= 0
            if i - m >= 0:
                total += weights[m] * np.sign(ac[m]) * x[i - m]
        grad_soft[i] = total / beta
    
    # Full gradient of objective = soft_max / s^2
    # d/dx_i [soft_max / s^2] = grad_soft[i]/s^2 - 2*soft_max*x[i]/s^3
    grad_obj = grad_soft / (s * s) - 2.0 * soft_max_val * x / (s ** 3)
    
    return obj, grad_obj

def _objective_and_grad(x, beta=50.0):
    """Compute objective and gradient for L-BFGS."""
    obj, grad = _soft_max_and_grad(x, beta)
    return obj, grad

def _spectral_init():
    """Generate initial signal with Gaussian spectrum (zero phase)."""
    k = np.arange(N)
    kf = k / N
    mag = np.exp(-2.0 * kf**2)
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    return x

def _spectral_phase_optimize(x0, max_iter=200, beta=50.0):
    """Optimize the phase of the spectrum to minimize the objective."""
    # Work in the frequency domain: fix |X[k]|, optimize phase.
    # This is a non-convex problem but L-BFGS can help.
    
    L_fft = 1
    while L_fft < 2*N-1:
        L_fft *= 2
    
    X0 = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X0)
    phase0 = np.angle(X0)
    
    # Optimize phase: x = IFFT(mag * exp(i*phase))
    def _phase_obj(phase_flat):
        phase = phase_flat.reshape(-1)
        X = mag * np.exp(1j * phase)
        # Ensure Hermitian symmetry for real signal
        # rfft gives k=0 to L_fft//2
        # For real signal, X[k] and X[L_fft-k] are conjugates
        # rfft handles this automatically
        x = np.fft.irfft(X, n=L_fft)[:N]
        return _objective_only(x)
    
    # Start from current phase
    p0 = phase0.copy()
    try:
        result = minimize(_phase_obj, p0, method='L-BFGS-B', maxiter=max_iter,
                         options={'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        phase_opt = result.x.reshape(-1)
        X_opt = mag * np.exp(1j * phase_opt)
        x_opt = np.fft.irfft(X_opt, n=L_fft)[:N]
        return x_opt, result.fun
    except Exception:
        return x0, _objective_only(x0)

def _amplitude_optimize(x0, max_iter=100):
    """Optimize the amplitude of the spectrum."""
    L_fft = 1
    while L_fft < 2*N-1:
        L_fft *= 2
    
    X0 = np.fft.rfft(x0, n=L_fft)
    mag0 = np.abs(X0)
    phase0 = np.angle(X0)
    
    def _amp_obj(log_mag):
        mag = np.exp(log_mag)
        X = mag * np.exp(1j * phase0)
        x = np.fft.irfft(X, n=L_fft)[:N]
        return _objective_only(x)
    
    p0 = np.log(mag0 + 1e-10)
    try:
        result = minimize(_amp_obj, p0, method='L-BFGS-B', maxiter=max_iter,
                         options={'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        log_mag_opt = result.x
        mag_opt = np.exp(log_mag_opt)
        X_opt = mag_opt * np.exp(1j * phase0)
        x_opt = np.fft.irfft(X_opt, n=L_fft)[:N]
        return x_opt, result.fun
    except Exception:
        return x0, _objective_only(x0)

def _peak_targeted_polish(x, num_peaks=5, num_iters=200):
    """Identify dominant autoconvolution lags and apply targeted perturbations."""
    best_x = x.copy()
    best_val = _objective_only(x)
    rng = np.random.default_rng(123)
    
    for it in range(num_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        
        # Find top peaks
        top_indices = np.argsort(abs_ac)[-num_peaks:]
        
        # For each peak, try to reduce it by perturbing x
        for idx in top_indices:
            if _check_deadline():
                break
            
            # Compute which x elements contribute to this peak
            m = idx  # lag
            # ac[m] = sum_k x[k]*x[k+m]
            # To reduce |ac[m]|, perturb x in a way that reduces the correlation
            # Simple: add a random perturbation weighted by the contributing elements
            
            contrib = np.zeros(N)
            for k in range(N):
                if k + m < N:
                    contrib[k] += x[k + m]
                    if k - m >= 0:
                        contrib[k] += x[k - m]
            
            # Perturb opposite to the contribution
            direction = -np.sign(ac[m]) * contrib
            direction /= (np.max(np.abs(direction)) + 1e-10)
            
            for step in [0.01, 0.05, 0.1, 0.5]:
                x_trial = best_x + step * direction
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral initialization
    x_spec = _spectral_init()
    val_spec = _objective_only(x_spec)
    if val_spec < best_val:
        best_val = val_spec
        best_x = x_spec.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, max_iter=300, beta=50.0)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Amplitude optimization
    x_amp, val_amp = _amplitude_optimize(best_x, max_iter=150)
    if val_amp < best_val:
        best_val = val_amp
        best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Full L-BFGS optimization with soft-max objective
    try:
        result = minimize(_objective_and_grad, best_x, args=(80.0,),
                         method='L-BFGS-B', maxiter=500,
                         options={'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, num_peaks=8, num_iters=300)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent with structured perturbations
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, N)
    
    # Low-frequency sinusoidal perturbations
    freqs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    step_sizes = [0.01, 0.05, 0.1, 0.3]
    
    for freq in freqs:
        if _check_deadline():
            break
        for step in step_sizes:
            for sign in [1, -1]:
                perturbation = sign * step * np.sin(2 * np.pi * freq * t)
                x_trial = best_x + perturbation
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Sparse large-step perturbations
    for _ in range(500):
        if _check_deadline():
            break
        
        # Pick a random subset of indices
        num_sparse = rng.integers(5, 30)
        indices = rng.choice(N, size=num_sparse, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, 0.5, size=num_sparse)
        
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final high-precision polish with tighter L-BFGS
    try:
        result = minimize(_objective_and_grad, best_x, args=(100.0,),
                         method='L-BFGS-B', maxiter=300,
                         options={'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Phase 9: Final peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, num_peaks=10, num_iters=200)
    
    return best_x
# EVOLVE-BLOCK-END
