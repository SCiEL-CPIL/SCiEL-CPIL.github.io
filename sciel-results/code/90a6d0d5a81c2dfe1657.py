"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    # Circular cross-correlation via FFT
    # overlap[t] = sum_i h[i] * comp[(i+t) % N]
    # = circular cross-correlation of h and comp
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    # cross-correlation: ifft( H * conj(C) ) gives sum h[i] * conj(comp[i-t])
    # We want sum_i h[i] * comp[i+t] = ifft( conj(H) * C )[t] ... let's be careful
    # Actually: (h cross-comp)[t] = sum_i h[i] * conj(comp[i - t])
    # comp is real, so conj(comp) = comp
    # We want sum_i h[i] * comp[i + t]. Let j = i + t => i = j - t
    # = sum_j h[j - t] * comp[j] = (comp cross h)[t]... 
    # Let's just compute directly for small N using the FFT-based circular convolution.
    # sum_i h[i] * comp[i+t] for t=0..N-1
    # This is the circular cross-correlation. Using:
    # corr[t] = ifft( fft(h) * conj(fft(comp)) )  gives sum_i h[i] * conj(comp[i-t])
    # Since comp is real: sum_i h[i] * comp[i-t]
    # We want sum_i h[i] * comp[i+t]. Let t' = -t mod N: sum_i h[i] * comp[i - t'] where t' = N - t
    # So corr[(N - t) % N] = sum_i h[i] * comp[i+t]
    # Alternatively, just reverse: we can compute ifft(conj(fft(h)) * fft(comp)) 
    # = sum_i conj(h[i-t]) * comp[i] = sum_i h[i-t] * comp[i] (h real)
    # = sum_j h[j] * comp[j+t] with j = i-t => i = j+t
    # Yes! ifft(conj(H) * C) [t] = sum_j h[j] * comp[j+t]
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def construct():
    N = 100
    # Start with a two-level pattern: alternating blocks
    # We'll use a smooth multi-harmonic initial guess and optimize.
    
    np.random.seed(42)
    
    # Initial: a few harmonics with random phases and amplitudes
    h = np.full(N, 0.5)
    # Add harmonics k=1,2,3 with small amplitudes
    for k in [1, 2, 3, 5]:
        amp = 0.15 / k
        phase = np.random.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
    
    # Project to [0, 1] and fix sum
    def project(h):
        h = np.clip(h, 0.0, 1.0)
        # Adjust to make sum exactly N/2
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        # Re-clip and iterate a few times
        for _ in range(50):
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h
    
    h = project(h)
    
    # Optimized: use scipy if available, otherwise manual gradient descent
    try:
        from scipy.optimize import minimize
        import scipy
        
        def objective(x):
            h = project(x)
            return _overlap_max(h, N)
        
        # Use Nelder-Mead for derivative-free optimization
        res = minimize(objective, h, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        h = project(res.x)
    except Exception:
        # Fallback: manual projected gradient descent
        lr = 0.05
        for step in range(5000):
            h = project(h)
            # Compute overlap at all shifts
            comp = 1.0 - h
            H = np.fft.fft(h)
            C = np.fft.fft(comp)
            overlap = np.real(np.fft.ifft(np.conj(H) * C))
            # Gradient of max overlap w.r.t. h is tricky; use subgradient at argmax
            t_star = np.argmax(overlap)
            # overlap[t] = sum_j h[j] * comp[j+t]
            # d overlap[t] / d h[i] = comp[i+t] - h[i] (since comp = 1-h, d comp[i+t]/d h[i] = -1 if i+t==i... no)
            # Actually overlap[t] = sum_j h[j] * (1 - h[j+t])
            # = sum_j h[j] - sum_j h[j] h[j+t]
            # d/dh[i] = 1 - h[(i-t) mod N] - h[(i+t) mod N]  (from both terms where h[i] appears)
            # Wait: sum_j h[j] contributes 1 for j=i.
            # sum_j h[j] h[j+t]: h[i] appears when j=i (gives h[i+t]) and when j+t=i i.e. j=i-t (gives h[i-t])
            # So d/dh[i] of overlap[t] = 1 - h[(i+t)%N] - h[(i-t)%N]
            # But we're maximizing sum h = N/2 constraint... actually we just minimize overlap.
            grad = np.zeros(N)
            for i in range(N):
                grad[i] = 1.0 - h[(i + t_star) % N] - h[(i - t_star) % N]
            # But this is for the specific t_star. Since we want min max, subgradient approach.
            h = project(h - lr * grad)
    
    # Final projection to ensure exact constraints
    h = project(h)
    # Ensure sum is exactly N/2
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
