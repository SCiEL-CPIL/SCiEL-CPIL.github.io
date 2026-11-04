"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def construct():
    N = 100
    np.random.seed(42)
    
    # Start with a two-level pattern: alternating blocks
    # We'll use a smooth multi-harmonic initial guess and optimize.
    
    # Initial: a few harmonics with random phases and amplitudes
    h = np.full(N, 0.5)
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
