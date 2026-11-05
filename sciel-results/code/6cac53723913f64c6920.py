"""Rewritten program: spectral-parameterized step heights for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _objective(params):
    """
    params: [a1, a2, ..., aK, b1, b2, ..., bK]
    where a_i are real parts of poles (negative for stability), b_i are residues.
    We model h as a sum of exponentials: h[n] = sum_i b_i * exp(a_i * n)
    This gives a smooth, decaying (or growing if a_i > 0, but we want bounded) signal.
    
    Actually, for a step function on [-1/4, 1/4] with N intervals, the indices n=0..N-1
    correspond to positions. Let's use a cosine-like basis or simply optimize h directly
    but with a low-rank constraint? 
    
    Better approach: Use a parametric family that is known to minimize the peak-to-sum ratio.
    The seed np.ones(N) gives 2.0. 
    value = 2*N*max(conv)/sum(h)^2.
    For h=ones, conv[0]=N, max=N, sum=N, value=2*N*N/N^2=2.
    
    If we make h decay, say h[n] = r^n with r<1, then sum = (1-r^N)/(1-r).
    conv[k] = sum_{i=0}^{N-1-k} r^{2i+k} = r^k * (1-r^{2(N-k)})/(1-r^2).
    Max is at k=0: (1-r^{2N})/(1-r^2).
    value = 2*N * (1-r^{2N})/(1-r^2) / ((1-r^N)/(1-r))^2
    For r close to 1, let r=1-eps.
    This might not beat 2 easily.
    
    What if we use alternating signs? h[n] = (-1)^n.
    sum = 0 (invalid).
    h[n] = 1 for n even, -1 for n odd? sum=0.
    h[n] = cos(2*pi*n/N)? sum=0.
    
    We need sum(h) != 0.
    Let's try h[n] = 1 + A * cos(2*pi*n/N).
    If A is small, sum ~ N.
    conv will have a peak at 0 and a secondary peak at N/2.
    
    Let's use a generic optimization with a deadline.
    We will optimize a vector of parameters for a low-frequency cosine mixture.
    h[n] = sum_{k=0}^{K} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    To ensure sum != 0, we fix c_0 = 1 (DC component).
    We optimize the other coefficients to minimize the objective.
    """
    K = 10  # number of harmonics
    # params: c_1..c_K, d_1..d_K
    c = np.zeros(K)
    d = np.zeros(K)
    c[:] = params[:K]
    d[:] = params[K:]
    
    n = np.arange(N)
    h = np.ones(N)  # DC component
    for k in range(1, K + 1):
        h += c[k-1] * np.cos(2 * np.pi * k * n / N)
        h += d[k-1] * np.sin(2 * np.pi * k * n / N)
    
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    val = 2 * N * max_conv / (s * s)
    return val

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill
    
    # Initial guess: small random harmonics
    np.random.seed(42)
    x0 = np.random.randn(20) * 0.1
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Local optimization
    try:
        res = minimize(_objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    attempts = 0
    while time.time() < deadline:
        x0 = np.random.randn(20) * 0.5
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        attempts += 1
        if attempts > 50:
            break
    
    K = 10
    n = np.arange(N)
    h = np.ones(N)
    c = best_x[:K]
    d = best_x[K:]
    for k in range(1, K + 1):
        h += c[k-1] * np.cos(2 * np.pi * k * n / N)
        h += d[k-1] * np.sin(2 * np.pi * k * n / N)
    
    # Ensure sum is not zero and finite
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-6:
        return np.ones(N)
        
    return h

# EVOLVE-BLOCK-START
# (No changes needed to the entry point, construct is defined above)
# EVOLVE-BLOCK-END
