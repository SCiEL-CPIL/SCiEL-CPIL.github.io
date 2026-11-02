import time
import numpy as np
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def _compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    return np.max(conv) / (s * s)

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 130.0

    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    # Objective function for L-BFGS-B
    # We minimize the peak of the autoconvolution normalized by the square of the sum.
    # The peak is expected to be at the center index N-1.
    # To be safe, we can check the max over the whole convolution, but that's non-smooth.
    # Given the symmetry of autoconvolution and the "bump" nature, the center is the max.
    
    def objective(h):
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        conv = _fft_convolve(h)
        # Use the center value primarily, but if we want to be robust to shifts, 
        # we could use a smooth max. 
        # However, for non-negative h, the max is at the center.
        # Let's use the center value for smoothness.
        peak = conv[N - 1]
        return peak / (s * s)

    # Gradient of the objective
    # J = c_k / S^2 where S = sum(h)
    # dJ/dh_j = (1/S^2) * d(c_k)/d(h_j) - 2 * c_k / S^3 * dS/dh_j
    # dS/dh_j = 1
    # d(c_k)/d(h_j) = h_{k-j} + h_{j-(k-N+1)} (with bounds check)
    # For k = N-1:
    # d(c_{N-1})/d(h_j) = h_{N-1-j} + h_{j-(N-1-N+1)} = h_{N-1-j} + h_j
    # Wait, conv[k] = sum_i h_i h_{k-i}.
    # d/dh_j (sum_i h_i h_{k-i}) = h_{k-j} + h_{j-k+N-1} ?
    # Let's check indices.
    # Term where h_j is the first factor: i=j -> h_j h_{k-j}. Deriv is h_{k-j}.
    # Term where h_j is the second factor: k-i=j -> i=k-j.
    # Wait, sum is over i.
    # If k-i = j, then i = k-j.
    # So the term is h_{k-j} h_j. Deriv is h_{k-j}.
    # So d(c_k)/d(h_j) = h_{k-j} + h_{k-j} = 2 h_{k-j}?
    # No.
    # c_k = sum_{i=0}^{N-1} h_i h_{k-i} (assuming h_out=0).
    # If j is in [0, N-1]:
    # 1. h_j appears as h_i when i=j. Term: h_j h_{k-j}. Deriv: h_{k-j}.
    # 2. h_j appears as h_{k-i} when k-i=j => i=k-j. Term: h_{k-j} h_j. Deriv: h_{k-j}.
    # So if $0 \le k-j < N$, we get $2 h_{k-j}$?
    # Only if $j$ and $k-j$ are distinct indices in the sum?
    # If $j = k-j$, i.e., $2j=k$, then the term is $h_j^2$. Deriv is $2h_j$.
    # So generally, if $0 \le k-j < N$, derivative is $h_{k-j} + h_{k-j}$?
    # Let's check with small N.
    # N=2. h=[a,b].
    # c_0 = a^2. c_1 = 2ab. c_2 = b^2.
    # d(c_1)/d(a) = 2b. Formula: k=1, j=0. h_{1-0} = h_1 = b. h_{1-0}=b. Sum 2b. Correct.
    # d(c_1)/d(b) = 2a. Formula: k=1, j=1. h_{1-1} = h_0 = a. Sum 2a. Correct.
    # So d(c_k)/d(h_j) = 2 h_{k-j} if $0 \le k-j < N$, else 0.
    
    def objective_and_grad(h):
        s = np.sum(h)
        if s < 1e-12:
            return 0.0, np.zeros(N)
        
        conv = _fft_convolve(h)
        k = N - 1
        peak = conv[k]
        
        # Gradient of peak w.r.t h
        # g_peak[j] = 2 * h[k-j] if 0 <= k-j < N else 0
        # k = N-1.
        # k-j ranges from N-1 (j=0) to 0 (j=N-1).
        # So for all j in 0..N-1, k-j is in 0..N-1.
        # So g_peak = 2 * h[::-1]
        
        g_peak = 2.0 * h[::-1]
        
        # Gradient of J = peak / s^2
        # dJ = (s^2 * d(peak) - peak * 2s * ds) / s^4
        #    = (d(peak)/s^2) - (2 * peak * ds / s^3)
        # ds = sum(dh) = 1 for each component? No, dS/dh_j = 1.
        # So grad_J[j] = g_peak[j] / s^2 - 2 * peak / s^3
        
        grad = g_peak / (s * s) - (2.0 * peak / (s * s * s))
        
        return peak / (s * s), grad

    # Initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Tilted Uniform (Antisymmetric perturbation)
    # h_i = 1 + alpha * (i - (N-1)/2)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        x = np.arange(N) - (N - 1) / 2.0
        h = 1.0 + alpha * x / N # Scale alpha to be small relative to 1
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            initializations.append(h)
            
    # 3. Gaussian Bump (Symmetric)
    for sigma in [50, 100, 150]:
        x = np.arange(N) - (N - 1) / 2.0
        h = np.exp(-(x ** 2) / (2 * sigma ** 2))
        initializations.append(h)
        
    # 4. Cosine Bump
    for L in [100, 200, 300]:
        d = np.abs(np.arange(N) - (N - 1) / 2.0)
        h = np.where(d < L, np.cos(np.pi * d / (2 * L)), 0)
        initializations.append(h)

    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    for x0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = optimize.minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_cand = res.x
            h_cand = np.maximum(0, h_cand)
            
            # Re-evaluate score with the raw max to be sure
            sc = _compute_score(h_cand)
            
            if sc < best_score:
                best_score = sc
                best_h = h_cand.copy()
                
        except Exception:
            pass
            
    return best_h.tolist()
