import numpy as np
import time
import sys

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
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Represent the step function f as the squared magnitude of a complex signal x in the time domain.
    #    f = |x|^2.
    #    The autoconvolution of f is related to the autoconvolution of |x|^2.
    #    Actually, a more direct spectral approach:
    #    Let F = FFT(f). The autoconvolution peak is dominated by the DC and low frequencies.
    #    Minimizing the peak of the autoconvolution of f is equivalent to minimizing the peak of the 
    #    inverse FFT of |FFT(f)|^2.
    #    
    #    Alternative view:
    #    We want to minimize max_k (f * f)[k] / (sum f)^2.
    #    Let's use the property that for a non-negative f, the peak of f*f is minimized when f is as "flat" 
    #    as possible in the frequency domain? No, flat in time domain (uniform f) gives flat spectrum (delta at 0).
    #    Uniform f: f[i] = 1/N. FFT(f) has only DC component. f*f is triangular, peak at center.
    #    Score = 1/N.
    #    
    #    Can we do better?
    #    If we make f oscillatory, the convolution might have lower peak?
    #    Consider f = |cos(pi x)|^2 or similar.
    #    
    #    Let's try optimizing the frequency domain amplitudes directly.
    #    Let f = IFFT(|G|^2) where G is a complex vector.
    #    We want f >= 0.
    #    We want to minimize max(FFT(f) * FFT(f) in time domain) which is max(IFFT(|FFT(f)|^2)).
    #    This is circular.
    #    
    #    Let's stick to the time-domain signal x such that f = |x|^2.
    #    Then f*f = |x|^2 * |x|^2.
    #    In frequency domain: FFT(f*f) = FFT(f) * FFT(f).
    #    FFT(f) = FFT(|x|^2).
    #    This is not simple.
    #    
    #    Let's try a different structural change:
    #    Use a parametric family of functions that are known to have low autoconvolution peaks.
    #    For example, f(x) = (1 + cos(2*pi*k*x)) / 2.
    #    Or use a Gaussian-like shape? No, Gaussian has higher peak in convolution than uniform?
    #    Uniform is the minimum for the discrete L2-normalized peak of convolution?
    #    Actually, for discrete vectors, the uniform vector minimizes the max element of the convolution 
    #    normalized by sum^2.
    #    Proof: max(f*f) >= (sum f)^2 / (2N-1) by Cauchy-Schwarz or similar?
    #    Actually, sum(f*f) = sum(f^2) * N? No.
    #    sum_{k} (f*f)_k = (sum f)^2.
    #    There are 2N-1 terms in the full convolution.
    #    So max(f*f) >= (sum f)^2 / (2N-1).
    #    For N=600, 2N-1 = 1199.
    #    Uniform f: f[i]=1/N. (f*f)_k is triangular. Peak is N * (1/N)^2 = 1/N.
    #    1/N = 1/600 = 0.001666.
    #    Bound (sum f)^2 / (2N-1) = 1 / 1199 = 0.000834.
    #    So uniform is NOT optimal. We can potentially get down to 0.000834.
    #    The uniform distribution has a peak of 1/600. The theoretical lower bound is 1/1199.
    #    So there is room for improvement!
    #    
    #    How to achieve a flatter convolution?
    #    We need the energy of f*f to be spread out more evenly.
    #    This implies f should have a more "random" or "white noise" like structure in the time domain?
    #    Or perhaps a specific structured pattern.
    #    
    #    Let's try optimizing the time-domain vector f directly using gradient descent,
    #    but with a better initialization and a penalty for non-negativity.
    #    Objective: min max(f*f) subject to f >= 0.
    #    This is a non-smooth optimization problem (max).
    #    We can use a smooth approximation: min log-sum-exp(f*f / T) / (sum f)^2.
    #    Or simply minimize the sum of squares of the convolution values, which spreads the peak.
    #    min sum((f*f)^2) / (sum f)^4.
    #    This encourages the convolution to be flat.
    #    
    #    Let's try minimizing the L2 norm of the convolution.
    #    sum_k (f*f)_k^2 = ||f*f||_2^2.
    #    In frequency domain: ||f*f||_2^2 = ||FFT(f) * FFT(f)||_2^2 / N (depending on FFT normalization).
    #    Let F = FFT(f). Then f*f = IFFT(F * F).
    #    ||IFFT(F*F)||_2^2 = (1/N) ||F*F||_2^2 = (1/N) sum |F_k|^4.
    #    So minimizing the L2 norm of the convolution is equivalent to minimizing sum |F_k|^4.
    #    This is minimizing the kurtosis of the spectrum!
    #    We want the spectrum to be as flat as possible (uniform magnitude).
    #    If |F_k| is constant for all k, then f is a delta function (in time), which is not allowed (f must be non-negative step function, but delta is allowed if we interpret step heights).
    #    Wait, if f is a delta, f*f is a delta. Peak is 1. Sum is 1. Score 1. Bad.
    #    We want f to be "spread out" in time to spread out the convolution.
    #    
    #    Let's reconsider.
    #    Uniform f: F is delta. sum |F|^4 is large (concentrated).
    #    Random f: F is roughly uniform. sum |F|^4 is smaller.
    #    So minimizing sum |F|^4 (spectral flatness) should lead to a flatter convolution?
    #    Let's test this hypothesis.
    #    If we make f random, the convolution peak should be lower than uniform?
    #    Uniform peak: 1/N.
    #    Random f (i.i.d. exponential): f*f is sum of exponentials. Peak is roughly proportional to 1/sqrt(N)?
    #    Let's simulate mentally.
    #    If f is random, the convolution is a sum of N^2 terms. The peak is the maximum of a random walk.
    #    For large N, the peak of the convolution of two random vectors of length N with mean 1/N is roughly 
    #    (1/N^2) * (N * 1/N) * sqrt(N) = 1/N * sqrt(N) = 1/sqrt(N)?
    #    No, the mean of each term is 1/N^2. There are N terms at the center.
    #    The value at the center is sum_{i=1}^N f[i]f[N-i].
    #    If f[i] are i.i.d. with mean mu and variance sigma^2.
    #    E[center] = N mu^2.
    #    Var[center] = N sigma^4.
    #    If mu = 1/N, sigma^2 = 1/N^2 (for uniform, sigma=0).
    #    For exponential with mean 1/N, sigma^2 = 1/N^2.
    #    E[center] = N (1/N)^2 = 1/N.
    #    This is the same as uniform mean.
    #    But the variance is non-zero. The peak will be higher than the mean.
    #    So random f likely has a HIGHER peak than uniform.
    #    
    #    So uniform is likely local minimum for max peak.
    #    But the bound 1/(2N-1) suggests we can do better.
    #    How to achieve a flatter convolution?
    #    We need the energy to be spread over all 2N-1 bins.
    #    This requires f to have a specific structure.
    #    A "Barker sequence" or similar low-autocorrelation sequence?
    #    But f must be non-negative.
    #    
    #    Let's try a different approach:
    #    Minimize the peak of the convolution using subgradient descent on the time-domain vector.
    #    Objective: max_k (f*f)_k.
    #    Gradient of max is the one-hot vector at the argmax.
    #    Gradient of (f*f)_k w.r.t f[i] is sum_j f[j] * delta_{k, i+j} = f[k-i] (with appropriate bounds).
    #    So the subgradient is f[argmax - i] for the active constraint.
    #    
    #    Let's implement a simple projected subgradient descent.
    #    1. Initialize f to uniform.
    #    2. Compute conv = f*f.
    #    3. Find k_max = argmax(conv).
    #    4. Compute gradient g. g[i] = f[k_max - i] if valid, else 0.
    #    5. Update f = f - step * g.
    #    6. Project f to non-negative (clip to 0).
    #    7. Normalize f to have sum 1 (optional, since score is scale invariant).
    #    
    #    This should push the peak down.
    
    # Initialize
    f = np.ones(N) / N
    best_f = f.copy()
    best_score = compute_score(f)
    
    # Parameters
    step_size = 0.01
    decay = 0.99
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > TIME_LIMIT * 0.9:
            break
            
        # Compute convolution
        conv = np.convolve(f, f, mode='full')
        
        # Find peak
        k_max = np.argmax(conv)
        
        # Compute subgradient
        # grad[i] = f[k_max - i] if 0 <= k_max - i < N else 0
        # Note: conv index k corresponds to sum_{i+j=k} f[i]f[j].
        # d(conv[k])/df[i] = sum_j f[j] * (i+j=k) = f[k-i].
        grad = np.zeros(N)
        # Vectorized gradient computation
        # i ranges 0..N-1.
        # j = k_max - i.
        # Valid if 0 <= j < N.
        i_vals = np.arange(N)
        j_vals = k_max - i_vals
        valid = (j_vals >= 0) & (j_vals < N)
        grad[valid] = f[j_vals[valid]]
        
        # Update
        f_new = f - step_size * grad
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0)
        
        # Normalize to sum 1 to prevent drift
        s = np.sum(f_new)
        if s < 1e-15:
            break
        f_new = f_new / s
        
        # Check improvement
        score_new = compute_score(f_new)
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            # Maybe increase step size slightly if improving?
            # step_size *= 1.01
        else:
            # Decay step size
            step_size *= decay
        
        # Keep f as the current iterate for next step
        f = f_new
        
    # Final refinement: local coordinate descent on best_f
    # Try small perturbations on individual bins to see if we can shave off more.
    # This is expensive, so do it only if time permits.
    if time.time() - start_time < TIME_LIMIT * 0.8:
        # Simple coordinate descent
        for _ in range(100): # 100 passes
            if time.time() - start_time > TIME_LIMIT * 0.95:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT * 0.95:
                    break
                # Try increasing f[i]
                old_val = best_f[i]
                if old_val < 1e-15:
                    # Try small positive
                    best_f[i] = 1e-6
                    s = np.sum(best_f)
                    best_f = best_f / s
                    sc = compute_score(best_f)
                    if sc < best_score:
                        best_score = sc
                        improved = True
                    else:
                        best_f[i] = old_val
                        s = np.sum(best_f)
                        best_f = best_f / s
                else:
                    # Try scaling f[i] by 1+delta
                    for delta in [0.1, 0.01, -0.1, -0.01]:
                        new_val = old_val * (1 + delta)
                        if new_val < 0:
                            continue
                        best_f[i] = new_val
                        s = np.sum(best_f)
                        best_f = best_f / s
                        sc = compute_score(best_f)
                        if sc < best_score:
                            best_score = sc
                            improved = True
                            break
                        else:
                            best_f[i] = old_val
                            s = np.sum(best_f)
                            best_f = best_f / s
            
            if not improved:
                break
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
