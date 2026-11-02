"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 600
DEADLINE = 130.0  # safety margin before 150s kill

def _autoconv(h):
    """Compute full autoconvolution of h."""
    return fftconvolve(h, h[::-1], mode='full')

def _normalized_max_autoconv(h):
    """Compute normalized max of autoconvolution."""
    s = np.sum(h)
    if s <= 0:
        return 1e10
    ac = _autoconv(h)
    peak = np.max(ac)
    # Normalization: peak / (sum)^2 * N? Or peak / (sum * sum)?
    # For flat h=1: ac peak = N=600, sum=600, sum^2=360000
    # 600/360000 = 1/600 ≈ 0.001667
    # The score 1.0 corresponds to the seed, so let's use peak/(sum^2)
    return peak / (s * s)

def _logsumexp(x, scale=1.0):
    """Stable log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if m > 1e100:
        return m
    return m + np.log(np.sum(np.exp((x - m) / scale)) * scale) / scale

def _smooth_max_autoconv(h, beta=1.0):
    """Smooth surrogate for max of autoconvolution using log-sum-exp."""
    ac = _autoconv(h)
    s = np.sum(h)
    if s <= 0:
        return 1e10
    # Normalize first
    ac_norm = ac / (s * s)
    return _logsumexp(ac_norm, scale=beta)

def _project_nonnegative(h):
    """Project onto nonnegative orthant."""
    return np.maximum(h, 0.0)

def _optimize_from_init(h0, max_iter=2000, lr_init=0.01, beta=1.0, time_limit=120.0):
    """Projected gradient descent with log-sum-exp surrogate."""
    h = h0.copy().astype(np.float64)
    # Normalize to have sum = 1 for stability during optimization
    s = np.sum(h)
    if s <= 0:
        s = 1.0
    h = h / s
    
    best_score = _normalized_max_autoconv(h * s)  # keep track
    best_h = h.copy()
    
    t0 = time.time()
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        # Compute gradient of smooth objective
        ac = _autoconv(h)
        s = np.sum(h)
        ac_norm = ac / (s * s)
        
        # log-sum-exp gradient: softmax weights
        m = np.max(ac_norm)
        if m > 1e100:
            w = np.zeros_like(ac_norm)
            w[np.argmax(ac_norm)] = 1.0
        else:
            w = np.exp((ac_norm - m) / beta)
            w /= np.sum(w)
        
        # Gradient of sum(w_i * ac_i) / (s^2) w.r.t. h
        # d(ac_i)/d(h_j) = h[i-j] (for appropriate indexing)
        # ac = h * h[::-1] (full convolution)
        # d(ac_k)/d(h_j) = h[k-j] + h[k-j+N-1]... 
        # Actually for autoconv: ac[k] = sum_j h[j]*h[j+k-N+1] (with appropriate bounds)
        # Let's use: ac = fftconvolve(h, h[::-1], 'full')
        # d(ac[k])/d(h[j]) = h_rev[k-j] where h_rev = h[::-1]
        # = h[N-1-(k-j)] = h[N-1-k+j]
        
        h_rev = h[::-1]
        # grad_ac w.r.t. h: convolve w with h_rev appropriately
        # ac[k] = sum_j h[j] * h_rev[k-j]
        # d(ac[k])/d(h[j]) = h_rev[k-j]
        # So grad_h[j] = sum_k w[k] * h_rev[k-j] = (w * h_rev)[j] (cross-correlation)
        
        # w has length 2N-1, h_rev has length N
        # grad_h[j] = sum_k w[k] * h_rev[k-j] for k where 0<=k-j<N
        # This is cross-correlation of w and h_rev
        from scipy.signal import correlate
        grad_ac_h = correlate(w, h_rev, mode='valid')  # length N
        
        # Also account for the 1/(s^2) normalization and s in denominator
        # f = sum_k w[k] * ac[k] / s^2
        # df/dh[j] = (1/s^2) * grad_ac_h[j] + sum_k w[k] * ac[k] * d(1/s^2)/dh[j]
        # d(1/s^2)/dh[j] = -2/s^3
        # So: grad = (1/s^2) * grad_ac_h - 2*sum(w*ac_norm)/s^3 * s ... 
        # Actually: f = (sum_k w[k]*ac[k]) / s^2
        # df/dh_j = (grad_ac_h[j] * s^2 - sum_k w[k]*ac[k] * 2*s * 1) / s^4
        # = (grad_ac_h[j] * s - 2 * f * s) / s^2  ... let me redo
        # Let A = sum_k w[k]*ac[k], f = A/s^2
        # df/dh_j = (dA/dh_j * s^2 - A * 2*s * ds/dh_j) / s^4
        # = (grad_ac_h[j] * s^2 - A * 2*s) / s^4
        # = (grad_ac_h[j] * s - 2*A/s) / s^2
        # = (grad_ac_h[j] - 2*f) / s  (since A/s^2 = f, so A/s = f*s)
        # Wait: A/s = f*s, so 2*A/s^2 = 2*f
        # df/dh_j = (grad_ac_h[j]*s^2 - 2*A*s) / s^4 = (grad_ac_h[j] - 2*A/s^2 * s/s) ... 
        # Let me just compute numerically-friendly:
        # df/dh_j = grad_ac_h[j]/s^2 - 2*A/(s^3)
        # = (grad_ac_h[j] - 2*A/s) / s^2
        # = (grad_ac_h[j] - 2*f*s) / s^2
        
        A = np.dot(w, ac)
        grad = (grad_ac_h - 2.0 * A / s) / (s * s)
        
        # Step
        h_new = h - lr * grad
        h_new = _project_nonnegative(h_new)
        
        # Re-normalize to keep sum = 1
        s_new = np.sum(h_new)
        if s_new < 1e-10:
            break
        h_new = h_new / s_new
        
        # Evaluate
        score = _normalized_max_autoconv(h_new * N)  # scale back? 
        # Actually the normalization makes it scale-invariant, so:
        score = _normalized_max_autoconv(h_new)
        
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    return best_h, best_score

def construct():
    t_start = time.time()
    
    best_h = np.ones(N)
    best_score = _normalized_max_autoconv(best_h)
    
    # Try multiple initializations
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Flat (baseline)
    inits.append(np.ones(N))
    
    # 2. Raised cosine
    x = np.linspace(0, np.pi, N)
    inits.append(0.5 * (1 + np.cos(x)))
    
    # 3. Sine-squared
    x = np.linspace(0, 2*np.pi, N)
    inits.append(np.sin(x)**2 + 0.1)
    
    # 4. Random positive
    inits.append(rng.random(N) + 0.1)
    
    # 5. Bump (Gaussian-like)
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-4*x**2))
    
    # 6. Two bumps
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-20*(x-0.3)**2) + np.exp(-20*(x+0.3)**2))
    
    # 7. Linear ramp
    inits.append(np.linspace(0.1, 1.0, N))
    
    # 8. Triangular
    x = np.linspace(-1, 1, N)
    inits.append(1.0 - np.abs(x))
    
    # 9. Random with structure (low frequency)
    freq = rng.uniform(0.5, 3.0)
    x = np.linspace(0, 2*np.pi*freq, N)
    inits.append(0.5 + 0.3*np.sin(x) + 0.2*rng.random(N))
    inits[-1] = np.maximum(inits[-1], 0.01)
    
    # 10. Another random
    inits.append(rng.random(N) + 0.05)
    
    for i, h0 in enumerate(inits):
        if time.time() - t_start > DEADLINE - 10:
            break
        
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.maximum(h0, 0.0)
        
        # Try a couple of beta values
        for beta in [1.0, 0.1, 0.01]:
            if time.time() - t_start > DEADLINE - 5:
                break
            
            remaining = DEADLINE - time.time()
            if remaining < 2:
                break
            
            h_opt, score = _optimize_from_init(
                h0, max_iter=1000, lr_init=0.005, beta=beta,
                time_limit=min(remaining - 2, 15.0)
            )
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Final: make sure we return nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    # Return as list of floats
    return [float(x) for x in best_h]
