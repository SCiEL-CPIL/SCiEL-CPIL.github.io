"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0

    # The constant sequence attains the Cauchy-Schwarz lower bound:
    #   max_k |A_k| >= A_0 = sum x_i^2 >= (sum x_i)^2 / N,
    # so  max_k |A_k| / (sum x_i)^2  >=  1/N,
    # with equality for any constant sequence.  Start from all-ones.
    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    def _spectral_optimize(x0, max_iters=1000, lr=0.05, beta=200):
        """
        Optimize the continuous relaxation of x to minimize max |autoconv|.
        Uses a smooth surrogate: exp(beta * max |autoconv|).
        Operates on the power spectrum magnitude |F|.
        """
        n = len(x0)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        
        # Initialize from x0
        F = np.fft.rfft(x0, size)
        mag = np.abs(F)
        mag[mag < 1e-12] = 1e-12 # avoid log(0)
        
        # We optimize log(mag) to keep magnitudes positive and scale-invariant initially
        # Actually, let's optimize the phase and magnitude separately?
        # Simpler: optimize the continuous signal x directly using FFT-based gradients.
        
        # Let's stick to optimizing x directly but using FFT for fast autoconv.
        # Objective: min max_k |A_k|
        # Surrogate: min sum exp(beta * A_k) + sum exp(beta * -A_k)  (approx max)
        # Or just min sum exp(beta * |A_k|) is non-diff at 0.
        # Let's use min sum exp(beta * A_k) + sum exp(-beta * A_k) which is smooth.
        
        x = x0.astype(float).copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute Autoconvolution
            F = np.fft.rfft(x, size)
            A = np.fft.irfft(F * F, size)[:2 * n - 1]
            
            # Compute gradient of surrogate L = sum exp(beta A) + sum exp(-beta A)
            # dL/dA = beta * (exp(beta A) - exp(-beta A)) = 2 beta sinh(beta A)
            dL_dA = 2 * beta * np.sinh(beta * A)
            
            # dA/dx:
            # A = x * x (circular conv of size 2n-1? No, linear conv)
            # Linear conv A_k = sum x_i x_{k-i}
            # dA_k / dx_j = x_{k-j} + x_{j-k} (if indices valid, else 0)
            # This is essentially the circular convolution of x with itself if we pad correctly,
            # but for linear conv, the gradient is the circular convolution of x with x
            # if we consider the periodic extension? No.
            
            # Let's use the property:
            # FFT(A) = F^2
            # We need grad_x (sum dL_dA * A)
            # This is equivalent to:
            # grad_x = ifft( 2 * F * (fft(dL_dA) / 2) ) ? No.
            
            # Let's derive:
            # J = sum_k w_k A_k
            # A = conv(x, x)
            # J = <w, conv(x,x)> = <w * x, x> (cross-correlation)
            # dJ/dx = 2 * conv(w, x)
            
            # So we need to compute conv(dL_dA, x).
            # We can do this via FFT.
            # dL_dA is defined on [0, 2n-2]. Pad to size.
            # w = dL_dA
            # grad = 2 * ifft( fft(w, size) * fft(x, size) )[:n]
            
            W = np.fft.rfft(dL_dA, size)
            grad = 2 * np.fft.irfft(W * F, size)[:n]
            
            # Update
            x -= lr * grad
            
            # Normalize to keep integral roughly constant or just let it be?
            # The metric is invariant to scaling?
            # max |c A| / (c S)^2 = c max |A| / c^2 S^2 = (1/c) metric.
            # So larger c (scale) makes metric smaller!
            # Wait.
            # If we scale x by c, integral becomes c*S.
            # Autoconv becomes c^2 * A.
            # Max abs autoconv becomes c^2 * max|A|.
            # Ratio: c^2 max|A| / (c S)^2 = max|A| / S^2.
            # The metric is scale INVARIANT.
            # So we can fix the scale, e.g., ||x||_1 = 1 or ||x||_2 = 1.
            # Let's fix ||x||_2 = 1 to prevent drift.
            norm = np.linalg.norm(x)
            if norm > 1e-12:
                x /= norm
            
            # Project to real (it is already real)
            
        return x

    # Phase 1: Spectral Optimization from random starts
    num_starts = 5
    phase1_deadline = deadline - 30.0
    
    for i in range(num_starts):
        if time.time() > phase1_deadline:
            break
        
        # Random start
        if i == 0:
            x_start = np.ones(N)
        else:
            x_start = rng.standard_normal(N)
        
        x_opt = _spectral_optimize(x_start, max_iters=2000, lr=0.01, beta=100)
        
        # Project to binary/signs
        # Thresholding
        x_bin = np.sign(x_opt)
        x_bin[x_bin == 0] = 1.0
        
        # Check metric
        v = _metric(x_bin)
        if v < best_val:
            best_val = v
            best_x = x_bin.copy()
            
        # Also try continuous values scaled? No, task says "signed step heights".
        # Usually implies discrete or just real values? "sequence of signed step heights"
        # The seed used floats in gen_gauss.
        # But "signed autocorrelation... signed equal-width step functions"
        # The evaluator measures "raw value".
        # The seed program returns floats (best_x from gen_gauss is float).
        # However, "signed step heights" often implies +/- 1 or similar.
        # Let's assume real values are allowed, but binary is a strong candidate.
        # Let's evaluate the continuous x_opt too, scaled to have sum=1?
        # No, scale invariant.
        
        # Let's try to refine the binary solution with local search
        # Phase 2: Local search on best_x
        
    # Phase 2: Local Search (Sign Flips)
    x = best_x.copy()
    v = best_val
    improved = True
    max_inner = 20
    inner_iter = 0
    phase2_deadline = deadline - 5.0
    
    while improved and time.time() < phase2_deadline and inner_iter < max_inner:
        improved = False
        inner_iter += 1
        for i in range(N):
            if time.time() >= phase2_deadline:
                break
            old = x[i]
            x[i] = -old
            v2 = _metric(x)
            if v2 < v:
                v = v2
                improved = True
            else:
                x[i] = old
            if v < best_val:
                best_val = v
                best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
