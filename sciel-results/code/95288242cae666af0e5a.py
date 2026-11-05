"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT (O(N log N) instead of O(N^2))."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2*n-1]

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)

def _random_walk_init(rng, N):
    """Generate random walk with negative drift as initialization."""
    drift = -0.01
    noise_scale = 0.5
    h = np.zeros(N)
    for i in range(N):
        h[i] = h[i-1] + drift + noise_scale * rng.standard_normal() if i > 0 else noise_scale * rng.standard_normal()
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Polynomial bases
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. DCT-shaped initializations
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm
        initializations.append(h0)
    
    # 7. Random walk initializations
    for _ in range(5):
        h0 = _random_walk_init(rng, N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 8. Sinusoidal parameterized initializations
    i = np.arange(N)
    for _ in range(10):
        f = rng.uniform(1.0, N/2.0)
        amp = rng.uniform(-2.0, 2.0)
        phase = rng.uniform(0, 2*np.pi)
        baseline = rng.uniform(-1.0, 1.0)
        h0 = baseline + amp * np.cos(2*np.pi*f*i/N + phase)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 9. Multi-sinusoidal initializations
    for _ in range(5):
        h0 = np.zeros(N)
        for _ in range(3):
            f = rng.uniform(1.0, N/4.0)
            amp = rng.uniform(-1.5, 1.5)
            phase = rng.uniform(0, 2*np.pi)
            h0 += amp * np.cos(2*np.pi*f*i/N + phase)
        h0 += 0.5
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # Optimize each initialization with L-BFGS-B on soft-max objective
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        val_curr = _objective_hard(h_curr)
        
        # Try L-BFGS-B optimization with different beta values
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(_soft_max_obj, h_curr,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
                val_new = _objective_hard(res.x)
                if val_new < val_curr:
                    h_curr = res.x
                    val_curr = val_new
            except Exception:
                pass
        
        # Update best
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0, 500.0, 1000.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
