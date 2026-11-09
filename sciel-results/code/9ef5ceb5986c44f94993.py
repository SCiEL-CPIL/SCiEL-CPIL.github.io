import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _soft_max_abs_ac(f, beta=50.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient) using vectorized computation.
    """
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) / beta + max_ac
    
    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac)
    
    # Vectorized gradient: grad[i] = 2 * sum_k weights[k] * sign(ac[k]) * f[k-i] for valid k-i
    # ac[k] = sum_j f[j]*f[k-j], so d|ac[k]|/df[i] = sign(ac[k]) * 2 * sum_j delta(j,i) f[k-j] = sign(ac[k]) * 2 * f[k-i]
    # grad[i] = 2 * sum_k weights[k] * sign(ac[k]) * f[k-i] where 0<=k-i<N
    # This is a correlation-like operation: grad = 2 * (weights * signs) convolved with f (reversed)
    # Actually: grad[i] = 2 * sum_k w[k]*sgn[k]*f[k-i] for k-i in [0,N)
    # Let g[k] = weights[k] * signs[k] for k in [0, 2N-2)
    # grad[i] = 2 * sum_k g[k] * f[k-i] where k-i in [0,N-1], i in [0,N-1]
    # This is cross-correlation of g with f, but with specific indexing.
    # grad[i] = 2 * sum_{k=i}^{i+N-1} g[k] * f[k-i] for i in [0,N-1]
    # = 2 * (g * f)[i] in the sense of: sum over j of g[i+j]*f[j]
    
    # Use FFT for this: grad[i] = 2 * sum_j g[i+j] * f[j]
    # This is like a cross-correlation. Let's use direct FFT approach.
    # Alternatively, for N=400 and 2N-1=799, a vectorized loop over i is fine:
    # grad[i] = 2 * np.dot(g[i:i+N], f)
    
    g = weights * signs
    # Vectorize: for each i, grad[i] = 2 * dot(g[i:i+N], f)
    # Use sliding_window_view or manual
    # For N=400, this is 400 dots of length 400 = 160K ops, very fast
    grad = np.array([2.0 * np.dot(g[i:i+N], f) for i in range(N)])
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if s == 0 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    # d/d f log(smax) = grad_smax / smax
    # d/d f (-2 log|s|) = -2 * ones / s
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = np.inf
    
    ac = _autoconv(best)
    s = np.sum(best)
    if abs(s) > 1e-12:
        best_score = np.max(np.abs(ac)) / (s * s)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips (more aggressive)
        n_flips = np.random.randint(1, 6)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        ac = _autoconv(candidate)
        s = np.sum(candidate)
        if abs(s) > 1e-12:
            score = np.max(np.abs(ac)) / (s * s)
            if score < best_score:
                best = candidate
                best_score = score
        else:
            continue
        
        # Try small random perturbations
        if it % 3 == 0:
            candidate = best + np.random.randn(N) * 0.01
            ac = _autoconv(candidate)
            s = np.sum(candidate)
            if abs(s) > 1e-12:
                score = np.max(np.abs(ac)) / (s * s)
                if score < best_score:
                    best = candidate
                    best_score = score
    
    return best, best_score

def _spectral_init(seed=0):
    """
    Spectral shaping initialization: design frequency response to suppress
    large autoconvolution peaks, then invert to time domain.
    """
    rng = np.random.RandomState(seed)
    # Design a frequency response that is flat (white) to minimize peaks
    # Start with random phase, flat magnitude, then shape to reduce peaks
    n_freqs = N_FFT // 2 + 1
    # Random initial phases
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    # Start with flat magnitude
    mags = np.ones(n_freqs)
    mags[0] = 0  # zero DC to ensure sum is controlled
    
    F = mags * np.exp(1j * phases)
    # Make it conjugate symmetric for real signal
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1]) if n_freqs > 1 else 0
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    # Iteratively suppress peaks in autoconvolution
    for _ in range(50):
        ac = _autoconv(f)
        s = np.sum(f)
        if abs(s) < 1e-12:
            break
        # Find where peaks are
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        # Reduce contribution at this lag by adjusting signal
        # Simple approach: subtract a small multiple of the signal shifted
        shift = peak_idx % N
        if shift == 0:
            f -= 0.01 * f
        else:
            # Adjust to reduce correlation at this lag
            f -= 0.005 * np.roll(f, shift)
    
    # Normalize
    s = np.sum(f)
    if abs(s) > 1e-12:
        f = f / s  # scale so sum = 1
    else:
        f = rng.randn(N)
    
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_f = None
    best_score = np.inf
    
    # Initial guesses - more restarts since gradient is now fast
    n_restarts = 12
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.random.randn(N)
        elif r == 2:
            f0 = np.random.choice([-1.0, 1.0], size=N)
            if np.sum(f0) < 0:
                f0 = -f0
        elif r == 3:
            # Chirp
            t = np.linspace(0, 10, N)
            f0 = np.cos(t**2)
        elif r == 4:
            # Alternating blocks
            f0 = np.ones(N)
            f0[::2] = -1.0
        elif r == 5:
            # Random with structure
            f0 = np.random.randn(N // 2)
            f0 = np.repeat(f0, 2)
        elif r == 6:
            # Sine wave
            t = np.linspace(0, 4 * np.pi, N)
            f0 = np.sin(t)
        elif r == 7:
            # Spectral init 1
            f0 = _spectral_init(seed=r * 100 + 1)
        elif r == 8:
            # Spectral init 2
            f0 = _spectral_init(seed=r * 100 + 2)
        elif r == 9:
            # Spectral init 3
            f0 = _spectral_init(seed=r * 100 + 3)
        elif r == 10:
            # Random with block structure
            block_size = np.random.randint(5, 20)
            n_blocks = N // block_size
            blocks = np.random.randn(n_blocks)
            f0 = np.repeat(blocks, block_size)[:N]
        else:
            f0 = np.random.randn(N)
        
        # Optimization - more iterations since gradient is fast
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            if res.success or res.fun < best_score:
                candidate = res.x
                # Calculate actual score
                ac = _autoconv(candidate)
                s = np.sum(candidate)
                if abs(s) > 1e-12:
                    score = np.max(np.abs(ac)) / (s * s)
                    if score < best_score:
                        best_f = candidate
                        best_score = score
                        
        except Exception as e:
            pass
        
        # Local search polish
        if best_f is not None and time.time() < deadline - 5.0:
            polished, polished_score = _local_search(best_f, deadline - 5.0, max_iters=20000)
            if polished_score < best_score:
                best_f = polished
                best_score = polished_score
    
    # If no valid result, return ones
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
