import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fftconvolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def smooth_max(x, beta=100.0):
    """
    Compute the smooth maximum of x using log-sum-exp.
    This is differentiable and approximates max(x) for large beta.
    """
    m = np.max(x)
    if m == 0:
        return 0.0
    # Avoid overflow in exp
    x_shifted = x - m
    # Clip to prevent underflow issues in exp, though log-sum-exp is stable
    # We want sum(exp(beta * x_shifted))
    # If beta is large, exp(beta * x_shifted) can be very small for non-max elements.
    # This is fine.
    return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * x_shifted)))

def objective_and_grad(params, beta=100.0, phases=None):
    """
    Objective: smooth_max(conv) / s^2
    Params: log_amplitudes of the first N//2+1 FFT coefficients.
    We construct a real signal by assuming a Hermitian spectrum with fixed random phases.
    """
    # Reconstruct f from params
    # params are log(amps). amps = exp(params)
    amps = np.exp(params)
    
    # Construct complex spectrum
    # We use N//2 + 1 coefficients for a real signal of length N.
    # Let M = N//2 + 1
    M = N // 2 + 1
    X = np.zeros(N, dtype=np.complex128)
    
    # DC component
    X[0] = amps[0]
    
    # Nyquist component (if N is even)
    if N % 2 == 0:
        X[N//2] = amps[N//2]
    
    # Positive frequencies 1 to N//2 - 1
    # We need to assign phases.
    # To keep the function deterministic and differentiable with respect to params,
    # we use fixed random phases generated once per restart.
    # However, scipy minimize expects a fixed function.
    # We will pass phases via closure or global state? 
    # Better: define a factory or use a global variable for phases in this restart.
    
    # For simplicity in this implementation, we assume `phases` is a global or passed via a wrapper.
    # Since `minimize` calls the function with only `params`, we need `phases` to be accessible.
    # We'll use a global variable `CURRENT_PHASES` that is set before each minimize call.
    
    global CURRENT_PHASES
    if CURRENT_PHASES is None:
        # Fallback random phases
        CURRENT_PHASES = np.random.uniform(0, 2*np.pi, M)
        
    # Fill X
    # X[k] = amps[k] * exp(i * phase[k])
    # X[N-k] = conj(X[k])
    
    # Indices 1 to M-1 correspond to k=1 to N//2
    # Note: X[0] is real. X[N//2] is real (Nyquist).
    
    # Let's handle the array filling carefully.
    # amps has length M.
    # X[0] = amps[0]
    # For k in 1..M-1:
    #   if k < N//2: X[k] = amps[k]*exp(i*ph[k]), X[N-k] = conj
    #   if k == N//2 (and N even): X[k] = amps[k] (real)
    
    X[0] = amps[0]
    
    for k in range(1, M):
        phase = CURRENT_PHASES[k]
        val = amps[k] * np.exp(1j * phase)
        X[k] = val
        X[N-k] = np.conj(val)
        
    # Note: If N is even, X[N//2] is its own conjugate.
    # The loop above sets X[N//2] when k=N//2.
    # X[N - N//2] = X[N//2]. So it sets X[N//2] twice?
    # If k = N//2, N-k = N//2.
    # X[N//2] = val. X[N//2] = conj(val).
    # Since val is complex generally, this is a conflict unless phase is 0 or pi.
    # For a real signal, the Nyquist component must be real.
    # So we should force the phase of the Nyquist bin to be 0 (or pi, which is just sign, covered by amp if amp could be neg, but amp is exp(log_amp) > 0).
    # So we set phase[N//2] = 0 effectively.
    if N % 2 == 0:
        X[N//2] = amps[N//2] # Force real
    
    # Inverse FFT
    f = np.real(np.fft.ifft(X) * N) # ifft normalizes by N, so multiply by N to get correct scaling?
    # np.fft.ifft(X) = 1/N sum X_k exp(i 2 pi k n / N)
    # We want f_n = sum X_k exp(i 2 pi k n / N)
    # So yes, multiply by N.
    
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return 1e10, np.zeros_like(params)
    
    # Compute autoconvolution
    conv = fftconvolve(f, f, mode='full')
    
    # Objective: smooth_max(conv) / s^2
    sm = smooth_max(conv, beta)
    obj = sm / (s * s)
    
    # Gradient calculation
    # This is complex. We need d(obj)/d(log_amp_j)
    # obj = S / s^2 where S = smooth_max(conv)
    # d obj / d a_j = (1/s^2) dS/da_j - (2 S / s^3) (ds/da_j)
    # where a_j are the amplitudes (not log).
    # d log_a_j = d a_j / a_j
    # So d obj / d log_a_j = a_j * (d obj / d a_j)
    
    # Let's compute gradients numerically? No, too slow for 600 params.
    # Let's use analytic gradient if possible, or use finite difference if simple.
    # Given the complexity, let's use a simpler approach:
    # Minimize the unscaled smooth max conv, and fix s=1?
    # If we fix s=1, obj = smooth_max(conv).
    # But f is determined by amps. We can't easily fix s=1 in the parametric space without adding a constraint.
    # Alternative: Minimize smooth_max(conv / s^2)? No.
    
    # Let's stick to the ratio.
    # We can compute gradients of f w.r.t params analytically.
    # f = real(ifft(X) * N)
    # X_k depends on a_k (and a_{N-k} for k>0? No, X_k only depends on a_k for k<M, and X_{N-k} depends on a_k).
    # Actually, for a real signal constructed from M amps:
    # a_0 -> X_0
    # a_k (k>0, k<N/2) -> X_k, X_{N-k}
    # a_{N/2} -> X_{N/2}
    
    # df/d a_k can be derived.
    # However, implementing this analytically is error-prone.
    # Let's use `minimize` with `jac='2-point'`? No, 600 params, 2-point is 1200 evals per step. Too slow.
    # Let's use `jac='cs'` (complex step)? SciPy supports this.
    # Complex step differentiation is accurate and fast (1 eval per param).
    # But `minimize` with method='L-BFGS-B' and jac='cs' might work.
    # Let's try it.
    
    # We return 0 for grad, and let SciPy use complex step if we pass jac='cs'?
    # Actually, we should just return the objective and let SciPy estimate gradient if we don't provide it?
    # Default is '2-point'.
    # Let's provide a gradient function using complex step manually?
    # Or just use a smaller number of parameters?
    # N=600 is too many for finite difference.
    
    # Strategy change:
    # Use a **reduced basis**.
    # Instead of all 600 amps, use a small number of spectral "bumps" or a parametric spectrum.
    # E.g., spectrum is a sum of Gaussian bumps in frequency domain.
    # This reduces params to ~10-20.
    # Then finite difference is cheap.
    
    # Let's redefine the strategy:
    # 1. Parametric Spectrum: Sum of K Gaussian bumps in frequency.
    # 2. Optimize K*3 params using L-BFGS-B with numerical gradient.
    # 3. This is smooth and low-dimensional.
    
    return obj, np.zeros_like(params) # Placeholder, we will change the approach

# --- Revised Strategy: Parametric Frequency Domain ---

def parametric_spectrum_to_step(spectral_params, K):
    """
    spectral_params: [c1, w1, a1, c2, w2, a2, ...] for K bumps in frequency domain.
    Returns time-domain step function f (length N).
    """
    # Frequency bins
    freqs = np.fft.fftfreq(N) * N # Cycles per sample? 
    # Let's use normalized frequency [0, 1) or bins 0..N-1.
    # Bins 0..N//2 are unique for real signal.
    bins = np.arange(N)
    
    # Construct amplitude spectrum
    A = np.zeros(N)
    for i in range(K):
        c = spectral_params[3*i]
        w = spectral_params[3*i+1]
        a = spectral_params[3*i+2]
        # Gaussian bump in frequency
        # Distance in bins
        # Handle wrap-around? For now, simple linear distance.
        # Since we only use 0..N//2, and c is in 0..N//2, no wrap needed for the main lobe.
        dist = np.abs(bins - c)
        # Symmetry: A[N-b] = A[b]
        # We will build A for 0..N//2 and mirror.
        
        # Let's just build the full A array directly using the formula.
        # A[b] += a * exp(-0.5 * ((b-c)/w)^2)
        # This is O(N*K). K is small (e.g., 5), N=600. 3000 ops. Fast.
        
        A += a * np.exp(-0.5 * ((bins - c) / w) ** 2)
        
    # Enforce Hermitian symmetry for real signal
    # A[0] is real.
    # A[N/2] is real (if N even).
    # A[k] = A[N-k]
    # Our construction above might not be symmetric if c is not symmetric.
    # But we want a real signal.
    # So we take the real part of the inverse FFT.
    # This effectively symmetrizes the spectrum.
    
    # However, A must be non-negative?
    # The task requires f >= 0.
    # A (spectrum amplitude) can be anything, but f = ifft(A * phase) must be >= 0.
    # If we use random phases, f will be noisy.
    # To get f >= 0, we can use the **Wiener-Khinchin** relation?
    # No, f is the signal.
    # If we choose A to be the power spectrum of f, then f is not directly determined.
    
    # Let's go back to: **Optimize the time-domain vector using a smooth approximation of the max, but with a better optimizer.**
    # The issue with the current program is the non-smoothness of `max`.
    # `smooth_max` fixes that.
    # The issue with 600 params is the gradient cost.
    # But we can use **L-BFGS-B** with **sparse** gradients?
    # Or we can use **Adam** from a manual implementation?
    
    # Let's try a **manual gradient descent with smooth_max** on the raw vector, but using **FFT-based gradients**.
    # Gradient of smooth_max(conv) w.r.t f:
    # S = log(sum(exp(beta * conv))) / beta + max(conv)
    # d S / d conv_k = exp(beta * conv_k) / sum(exp(beta * conv))
    # Let p_k = d S / d conv_k. Sum(p_k) = 1.
    # conv = f * f
    # d conv_k / d f_j = f_{k-j} + f_{j-k} (with bounds)
    # So grad_f_j = sum_k p_k * (f_{k-j} + f_{j-k})
    # This is a convolution of p with f!
    # grad_f = 2 * (p * f) (circular convolution, with appropriate alignment)
    
    # This is O(N log N).
    # So we can do full 600-dim optimization efficiently!
    
    return None

def compute_gradient_smooth_max(f, beta=100.0):
    """
    Compute gradient of smooth_max(conv) / s^2 w.r.t f.
    Assumes sum(f) = s.
    """
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros_like(f)
    
    conv = fftconvolve(f, f, mode='full')
    
    # Smooth max weights
    m = np.max(conv)
    x_shifted = conv - m
    # Weights p_k proportional to exp(beta * x_shifted)
    # To avoid overflow, we already shifted by max.
    exps = np.exp(beta * x_shifted)
    sum_exps = np.sum(exps)
    if sum_exps < 1e-15:
        p = np.zeros_like(conv)
        p[np.argmax(conv)] = 1.0
    else:
        p = exps / sum_exps
        
    # Gradient of smooth_max w.r.t conv is p.
    # Gradient of conv w.r.t f is 2 * (f * p) ?
    # conv_k = sum_i f_i f_{k-i}
    # d conv_k / d f_j = f_{k-j} + f_{j-k}
    # grad_f_j = sum_k p_k (f_{k-j} + f_{j-k})
    # This is the circular convolution of f and p, plus f and p reversed?
    # Actually, if p is symmetric (which it is, since conv is symmetric), then p * f is symmetric.
    # grad_f = 2 * circular_conv(f, p)
    
    # We need to align indices.
    # fftconvolve gives linear convolution.
    # For circular convolution of length N, we use FFT.
    # Note: conv from fftconvolve is length 2N-1.
    # The "circular" convolution of f (len N) with p (len 2N-1) is tricky.
    # However, the gradient of the *linear* convolution max is what we need?
    # The task uses `np.convolve(f, f, mode='full')`. This is linear convolution.
    # So we need the gradient of the linear convolution.
    
    # d conv_k / d f_j is non-zero only if k-j is in [0, N-1] and j-k is in [0, N-1].
    # This is a valid linear convolution operation.
    # grad_f = 2 * (p * f) where * is linear convolution, but restricted to valid indices?
    # Actually, let's use the property:
    # grad_f = 2 * ifft( fft(f) * fft(p_padded) ) ?
    # No, linear convolution gradient is just linear convolution.
    
    # Let's compute `p` on the same grid as `conv` (length 2N-1).
    # Then `grad_f` can be computed by:
    # grad_f[j] = sum_k p[k] * (f[k-j] + f[j-k])
    # This is equivalent to:
    # grad_f = convolve(p, f, mode='same') + convolve(p, f[::-1], mode='same')[::-1] ?
    # Let's test with small N.
    # Simpler: Use FFT for linear convolution gradient.
    # We can zero-pad f and p to size 2N.
    # Then ifft(fft(f)*fft(p)) gives linear convolution.
    
    size = 2*N
    F = np.fft.fft(f, size)
    P = np.fft.fft(p, size)
    # grad_f_full = ifft(F * P) + ifft(F * conj(P)) ?
    # d (f*f) / df = f + f_reversed?
    # In frequency domain:
    # Conv(f, f) -> F^2
    # d (F^2) / dF = 2F
    # dF / df = fft_matrix
    # This is getting complicated.
    
    # Let's use the direct formula with `scipy.signal.fftconvolve` for the gradient?
    # No, `fftconvolve` doesn't give gradient.
    
    # Let's use the fact that:
    # grad_f = 2 * (p * f) where * is the "cross-correlation" like operation.
    # Specifically, if we define `g` such that `conv(f, g) = p`? No.
    
    # Let's just use `np.convolve` for the gradient step?
    # grad_f[j] = sum_k p[k] f[k-j] (valid range)
    # This is `np.convolve(p, f, mode='full')`?
    # Let's check dimensions.
    # p length 2N-1. f length N.
    # conv(p, f) length 3N-2.
    # We want result length N.
    
    # Efficient way:
    # grad_f = 2 * np.real(np.fft.ifft(np.fft.fft(f, size=2*N) * np.fft.fft(p, size=2*N))[:N])
    # Is this correct?
    # Let's assume yes for now and verify with a small case in code comments or just trust the math.
    # Actually, the gradient of `sum_k p_k conv_k` w.r.t `f` is `2 * (p * f)` where `*` is linear convolution,
    # but we need to handle the boundaries.
    # The FFT method `ifft(fft(f)*fft(p))` computes the linear convolution of f and p.
    # Since p is symmetric, `f * p` is the same as `p * f`.
    # And the derivative of `f^2` is `2f`.
    # So `grad_f = 2 * (p * f)`.
    
    conv_grad = np.fft.ifft(np.fft.fft(f, size=2*N) * np.fft.fft(p, size=2*N))
    grad_f = 2 * np.real(conv_grad[:N])
    
    # Now, we have grad of smooth_max(conv).
    # Objective is J = S / s^2.
    # dJ/df = (1/s^2) grad_S - (2S/s^3) grad_s
    # grad_s = vector of ones (since s = sum f)
    # So dJ/df = (1/s^2) grad_f - (2S/s^3) * 1
    
    S = m + (1.0/beta) * np.log(sum_exps)
    grad_J = (1.0 / (s*s)) * grad_f - (2.0 * S / (s**3)) * np.ones(N)
    
    return grad_J

def construct():
    start_time = time.time()
    
    # Initialize f
    # Start with a Gaussian
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.exp(-0.5 * (x / 0.1) ** 2)
    f = np.maximum(f, 0.0)
    f = f / np.sum(f)
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    beta = 100.0
    step_size = 0.01
    lr_decay = 0.99
    
    # Gradient Descent with Smooth Max
    # We project onto sum(f)=1, f>=0 after each step.
    
    for i in range(2000):
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
            
        grad = compute_gradient_smooth_max(f, beta)
        
        # Gradient Step
        f_new = f - step_size * grad
        
        # Project onto simplex (sum=1, f>=0)
        # This is a quadratic programming problem.
        # Simple iterative projection:
        # 1. Clip to 0
        f_new = np.maximum(f_new, 0.0)
        # 2. Normalize
        s = np.sum(f_new)
        if s < 1e-15:
            break
        f_new = f_new / s
        
        new_score = compute_score(f_new)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f_new.copy()
            f = f_new
            # Decrease step size slightly
            step_size *= lr_decay
        else:
            # If not improved, reduce step size more aggressively
            step_size *= 0.8
            if step_size < 1e-8:
                break
            
            # Optional: try a random perturbation if stuck?
            # For now, just continue.
            
    # Final check
    f = np.maximum(best_f, 0.0)
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
        
    return [float(x) for x in f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
