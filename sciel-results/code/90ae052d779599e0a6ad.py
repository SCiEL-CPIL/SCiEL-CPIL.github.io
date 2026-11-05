"""
Strategy: Direct spectral control via Fourier coefficients.

The objective is:
    value = |2 * N * max_k (convolve(h,h)[k]) / (sum(h))^2|

Since convolve(h,h) is the autocorrelation, by the Wiener-Khinchin theorem
we can express max_k c_k in terms of the Fourier transform H(f) = FFT(h).
However, computing the max of the autocorrelation directly is what the evaluator does.
Instead of optimizing the time-domain vector h directly (which is high-dimensional),
we exploit the fact that the problem is scale-invariant and we can work with the
normalized shape.

Key Insight from previous attempts:
The seed np.ones(400) gives 2.0.
We need to drive the max autocorrelation down relative to the total energy.
If we make h have "flat" spectrum (white noise), the autocorrelation is close to
a delta function. But the max of the autocorrelation is c_0 = sum(h^2).
Wait, the evaluator uses np.max(np.convolve(h, h)).
convolve(h, h)[0] = sum(h_i * h_i) = sum(h^2).
Other lags k have c_k = sum h_i h_{i+k}.
Usually c_0 is the largest because it's the sum of squares.
Unless h has negative values that cause cancellation in c_0? No, c_0 is sum of squares, always positive and largest in magnitude for most signals unless the signal is anti-correlated with itself at lag 0? Impossible.
So max(conv) is almost always c_0 = sum(h^2), unless there is a large peak at some other lag k.
But wait, if h has negative values, c_k can be positive or negative.
np.max takes the maximum signed value.
So if all c_k are negative except c_0, max is c_0.
Can we make c_0 small? c_0 = sum(h^2).
The denominator is (sum(h))^2.
Let S = sum(h).
value = 2 * N * sum(h^2) / S^2.
By Cauchy-Schwarz, (sum(h))^2 <= N * sum(h^2).
So sum(h^2) / S^2 >= 1/N.
Therefore value >= 2 * N * (1/N) = 2.
So the theoretical lower bound is 2?
But the seed gives 2.0. And the current best score is 1.2250, which corresponds to value = 2.0 / 1.2250 = 1.632.
This implies that max(conv) is NOT c_0.
Why? Because max(conv) can be larger than c_0? No, c_0 is sum of squares.
Ah, np.convolve(h, h) for "full" mode.
c_0 in numpy convolve is the overlap at shift 0.
If h is centered, c_0 is the largest.
But if h is not centered?
The definition of autocorrelation usually assumes zero mean or specific centering.
Here h is just a vector.
convolve(h, h)[k] = sum_{i} h[i] h[i+k] (with boundary handling).
Actually, np.convolve(h, h, 'full') computes:
c[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] h[k-i] ? No.
Standard convolution: (a * b)[k] = sum_i a[i] b[k-i].
For autocorrelation, we usually want c[k] = sum_i h[i] h[i+k].
numpy.convolve(h, h) gives the convolution, which is related to autocorrelation by conjugation and reversal.
Specifically, convolve(h, h)[k] = sum_i h[i] h[k-i].
If we want sum_i h[i] h[i+k], that is the cross-correlation.
But the task says: `np.max(np.convolve(h, h))`.
Let's check the seed: h = ones(400).
convolve(ones, ones) is a triangle: [1, 2, ..., 400, ..., 2, 1].
Max is 400.
sum(h) = 400.
value = abs(2 * 400 * 400 / 400^2) = 2.0.
This matches.

Now, if we pick h such that sum(h) is large but sum(h^2) is small?
Impossible by C-S. sum(h^2) >= (sum h)^2 / N.
So if max(conv) = c_0 = sum(h^2), then value >= 2.
But we have scores > 1.0, meaning value < 2.0.
This implies that for the optimal h, max(conv(h,h)) is NOT c_0.
This happens if c_0 is small, but some other lag k has a large positive value.
How can c_0 = sum(h^2) be small? It's always non-negative.
How can c_k be larger than c_0?
c_k = sum_i h[i] h[k-i].
By Cauchy-Schwarz, |c_k| <= sqrt(sum h^2 * sum h^2) = sum h^2 = c_0.
So |c_k| <= c_0 for all k.
Thus c_k <= c_0.
So max(conv) must be c_0?
Wait. c_0 in numpy.convolve(h, h) is at index N-1?
Let's trace indices.
h has length N.
conv = np.convolve(h, h, 'full'). Length 2N-1.
conv[k] = sum_{i} h[i] h[k-i].
The term where i=k-i implies i=k/2? No.
The peak of the triangle for ones(400) is at index 399 (0-indexed).
conv[399] = sum_{i=0}^{399} h[i] h[399-i] = sum_{i=0}^{399} 1*1 = 400.
This corresponds to the "lag 0" autocorrelation if we consider the center of the convolution.
So max(conv) is indeed the sum of squares of h?
conv[399] = sum_{i=0}^{399} h[i] h[399-i].
If h is symmetric, this is sum h[i]^2.
If h is not symmetric, is it still sum h[i]^2?
No. sum_{i} h[i] h[N-1-i].
This is not sum h[i]^2 unless h is symmetric (h[i] = h[N-1-i]).
Ah! This is the key.
The evaluator computes `np.max(np.convolve(h, h))`.
The maximum of the convolution of h with itself is NOT necessarily the sum of squares.
It is the sum of products h[i] * h[N-1-i] at the peak?
Actually, for any h, the maximum of the autocorrelation (which is the convolution of h with reversed h) is at lag 0.
But `np.convolve(h, h)` is convolution, not correlation.
conv(h, h)[k] = sum_i h[i] h[k-i].
If we want correlation c[k] = sum_i h[i] h[i+k], that is conv(h, reverse(h)).
The task uses `np.convolve(h, h)`.
Let's check if max(conv(h,h)) can be less than sum(h^2).
Let h = [1, -1]. N=2.
conv = [1, 0, -1]. Max is 1.
sum(h^2) = 2.
sum(h) = 0. (Invalid, sum must not be 0).
Let h = [2, -1]. N=2.
conv = [4, -2, 1]. Max is 4.
sum(h) = 1. sum(h^2) = 5.
value = abs(2 * 2 * 4 / 1^2) = 16.
Seed value for N=2 would be 2.0?
h=[1,1]. conv=[1,2,1]. Max=2. sum=2. val = 2*2*2/4 = 2.
So [2,-1] gives 16, which is worse.
We want to minimize value.
We want max(conv(h,h)) to be small relative to sum(h)^2.
Since max(conv) <= sum(h^2) is not generally true?
Actually, max(conv(h,h)) can be much smaller than sum(h^2) if h has alternating signs?
Let h = [1, -1, 1, -1].
conv = [1, -2, 2, -2, 1] (approx, let's compute).
h = [1, -1, 1, -1].
conv[0] = 1*1 = 1.
conv[1] = 1*(-1) + (-1)*1 = -2.
conv[2] = 1*1 + (-1)*(-1) + 1*1 = 1+1+1 = 3?
Let's use numpy logic.
[1, -1, 1, -1] * [1, -1, 1, -1]
0: 1*1 = 1
1: 1*(-1) + (-1)*1 = -2
2: 1*1 + (-1)*(-1) + 1*1 = 3
3: 1*(-1) + (-1)*1 + 1*(-1) = -3
4: 1*1 + (-1)*(-1) + 1*1 = 3? No.
Indices:
i=0..3, j=k-i..
k=0: i=0, j=0 -> 1*1=1
k=1: i=0, j=1 -> 1*(-1); i=1, j=0 -> -1*1. Sum -2.
k=2: i=0, j=2 -> 1*1; i=1, j=1 -> -1*-1=1; i=2, j=0 -> 1*1. Sum 3.
k=3: i=0, j=3 -> 1*(-1); i=1, j=2 -> -1*1; i=2, j=1 -> 1*-1; i=3, j=0 -> -1*1. Sum -4.
k=4: i=1, j=3 -> -1*-1=1; i=2, j=2 -> 1*1; i=3, j=1 -> -1*-1=1. Sum 3?
Wait, np.convolve is full.
Let's just trust that we can shape the spectrum.
The "cross" idea from the directive suggests a structural change.
Instead of optimizing h directly, we can optimize the Fourier coefficients.
Let H[k] be the DFT of h.
h is real, so H[-k] = conj(H[k]).
sum(h) = H[0].
conv(h, h) DFT is H[k]^2.
We want to minimize max(conv) / sum(h)^2.
This is equivalent to minimizing the peak of the inverse DFT of H^2 relative to H[0]^2.
This is a problem of spectral shaping.
We can parameterize H[k] (for k=0..N/2) with amplitudes and phases.
Since h is real, phases are symmetric.
We can use a simple parameterization: h[i] = sum_{k} A[k] cos(2 pi k i / N + phi[k]).
Optimize A and phi.

Let's try a direct optimization on the time-domain vector but with a different objective landscape.
The previous approach used blocky candidates.
The "cross" might refer to crossing over solutions in a genetic algorithm?
Or maybe it refers to the cross term in the autocorrelation?
Let's stick to a robust optimization of the time-domain vector using L-BFGS-B on a smooth surrogate of the max.
The max is non-differentiable. We use a log-sum-exp approximation.
However, the previous program already did this and got stuck.
The stagnation signal suggests a new representation.

Let's try optimizing the Fourier amplitudes directly.
We fix the DC component (sum h) to be 1 (scale invariance).
We optimize the AC components to flatten the autocorrelation peak.
Actually, we want to minimize the peak of the autocorrelation.
The autocorrelation is the inverse DFT of |H[k]|^2 (if we were doing correlation).
But we are doing convolution.
The DFT of conv(h, h) is H[k]^2.
So the peak of conv(h, h) is the max of the inverse DFT of H[k]^2.
We want to minimize max(IDFT(H^2)) subject to H[0] = 1.
This is a convex optimization problem if we ignore the max? No, max is convex.
Minimizing a convex function is good.
We can use L-BFGS-B on the amplitudes A[k] and phases phi[k].
Dimensions: N/2 + 1 complex numbers = N+2 real numbers.
This is 400 dimensions. Still high.

Alternative: Use a sparse representation in the Fourier domain.
Keep only the first K frequencies.
h[i] = A[0] + sum_{k=1}^{K} 2 A[k] cos(2 pi k i / N).
Sum h = A[0] * N.
We fix A[0] = 1/N so sum h = 1.
Optimize A[1]..A[K].
This reduces dimension to K.
We can try K=10, 20, 50.
Objective: value(h).
Use Nelder-Mead or L-BFGS-B.

Let's implement this "Sparse Fourier" approach.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin

def _evaluate(h):
    """Compute the exact task value."""
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _fourier_to_time(amplitudes, n_freqs):
    """
    Convert a vector of amplitudes (for cosine terms) to a time-domain signal.
    amplitudes[0] is the DC component (divided by N to keep sum=1? No, we handle scaling).
    We define h[i] = A[0] + sum_{k=1}^{K} 2 * A[k] * cos(2 * pi * k * i / N)
    To ensure sum(h) = 1, we note that sum_{i} cos(2 pi k i / N) = 0 for k != 0.
    So sum(h) = N * A[0].
    Thus we set A[0] = 1 / N.
    The amplitudes vector provided will be for k=1..K.
    """
    i = np.arange(N)
    h = np.full(N, 1.0 / N)  # DC component
    for k in range(1, n_freqs + 1):
        h += 2.0 * amplitudes[k-1] * np.cos(2.0 * np.pi * k * i / N)
    return h

def _fourier_objective(amplitudes, n_freqs):
    """
    Objective function for optimization.
    amplitudes: array of length n_freqs (for k=1..n_freqs).
    """
    h = _fourier_to_time(amplitudes, n_freqs)
    return _evaluate(h)

def construct():
    start_time = time.time()
    
    best_val = np.inf
    best_h = None
    
    # Fallback: Seed solution
    h_seed = np.ones(N)
    val_seed = _evaluate(h_seed)
    best_val = val_seed
    best_h = h_seed.copy()
    
    # Strategy: Optimize sparse Fourier coefficients
    # We try different numbers of frequencies K.
    # For each K, we run Nelder-Mead optimization.
    
    # We need to initialize the amplitudes.
    # Random initialization with small values.
    
    rng = np.random.RandomState(42)
    
    # Try different K values
    K_list = [5, 10, 20, 50, 100]
    
    for K in K_list:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize amplitudes
        # Start with zero or small random
        init_amp = rng.normal(0, 0.01, K)
        
        # Run Nelder-Mead
        # We might need to scale the objective or use a smooth surrogate if NM struggles with non-differentiability.
        # But NM handles non-differentiable functions well.
        
        try:
            # Use a time limit for this specific optimization
            t_limit = 15.0
            if time.time() - start_time + t_limit > DEADLINE - 5:
                t_limit = max(1.0, DEADLINE - 5 - (time.time() - start_time))
            
            # We wrap the objective to enforce time limit? No, NM doesn't take a time limit directly.
            # We can just run it and check time afterwards, or use a callback?
            # NM doesn't have a callback for time.
            # We'll just run it with a maxiter limit.
            
            res = minimize(
                lambda x: _fourier_objective(x, K),
                init_amp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                h_opt = _fourier_to_time(res.x, K)
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break

    # If we have time, try a local polish on the best h found using random perturbations
    if best_h is not None and time.time() - start_time < DEADLINE - 5:
        # Simple random search around the best solution
        # This might help escape local minima of the sparse Fourier representation
        rng2 = np.random.RandomState(123)
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 2:
                break
            # Add small Gaussian noise
            h_pert = best_h + rng2.normal(0, 0.01, N)
            val_pert = _evaluate(h_pert)
            if val_pert < best_val:
                best_val = val_pert
                best_h = h_pert.copy()

    # Final validity check
    if best_h is None or not np.isfinite(best_val):
        best_h = h_seed
        best_val = val_seed
        
    return best_h

# Test locally if run directly
if __name__ == "__main__":
    h = construct()
    print(_evaluate(h))
