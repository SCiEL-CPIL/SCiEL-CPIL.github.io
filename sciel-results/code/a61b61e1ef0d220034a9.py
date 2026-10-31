"""Strategy: Spectral Shaping via Fourier Coefficient Optimization.

The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale invariant.
We fix ||h||_1 = 1.
Maximizing R is equivalent to maximizing ||h||_2^2 / ||h*h||_inf.
Note that ||h*h||_inf >= (h*h)[0] = ||h||_2^2 (since h is non-negative, the peak is at least the DC component of the convolution).
Actually, for non-negative h, (h*h)[0] = sum h_i^2 = ||h||_2^2.
So ||h*h||_inf >= ||h||_2^2.
Thus R <= 1. Wait, the problem says "maximize the ratio".
Let's check the definition.
Ratio = (sum h_i^2) / (sum h_i * max(conv(h,h))).
If h is a delta function, sum h^2 = 1, sum h = 1, conv max = 1. Ratio = 1.
If h is uniform (1/50), sum h^2 = 50*(1/50)^2 = 1/50. sum h = 1. conv max = 1/50 (center peak is sum (1/50)^2 * 50 = 1/50). Ratio = (1/50) / (1 * 1/50) = 1.
Wait, if h is uniform, conv(h,h) is triangular. Max is at center.
h_i = 1/N.
(h*h)_k = sum_j h_j h_{k-j}.
For k=0 (index 0 in full conv? No, indices 0..2N-1).
Let's use 0-indexed h[0..N-1].
conv[0] = h[0]^2.
conv[N-1] = sum_{i=0}^{N-1} h[i]^2 = ||h||_2^2.
Since h >= 0, all terms in conv are non-negative.
The maximum of conv is at least ||h||_2^2.
So Ratio <= ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
If we fix ||h||_1 = 1, Ratio <= 1.
But the current program scores 1.15. How?
Ah, "supremum norm of the autoconvolution".
Maybe the convolution is normalized differently? Or my assumption about the max location is wrong?
If h is non-negative, the convolution is a "sliding sum of products".
The peak of the autoconvolution of a non-negative function is always at the lag 0 (or center, depending on indexing).
For discrete: (h*h)[k] = sum_j h[j] h[k-j].
If h is supported on 0..N-1, the max is at k=N-1 (center of full conv) where the overlap is maximal?
No, if h is non-negative, the sum is maximized when the two copies of h overlap as much as possible.
For non-negative h, the maximum of the autoconvolution is indeed (h*h)[N-1] = sum h_i^2.
Wait, if h = [1, 0, 0], conv = [1, 0, 0, 0, 0]. Max = 1. sum h^2 = 1. Ratio = 1/(1*1) = 1.
If h = [1, 1], conv = [1, 2, 1]. Max = 2. sum h^2 = 2. Ratio = 2/(2*2) = 0.5.
If h = [1, 1, 1], conv = [1, 2, 3, 2, 1]. Max = 3. sum h^2 = 3. Ratio = 3/(3*3) = 1/3.
It seems for non-negative h, the ratio is always <= 1?
Let's re-read the task carefully.
"maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
If h is non-negative, ||h*h||_inf >= ||h||_2^2.
So R <= 1/||h||_1.
If we don't fix ||h||_1, R is scale invariant?
If h -> c h, ||ch||_2^2 = c^2 ||h||_2^2. ||ch||_1 = c ||h||_1. ||ch*ch||_inf = c^2 ||h*h||_inf.
R(c h) = c^2 ||h||_2^2 / (c ||h||_1 * c^2 ||h*h||_inf) = (1/c) R(h).
So R depends on scale!
"Choose 50 nonnegative heights".
If I can choose the scale, I can make R arbitrarily large by making ||h||_1 small?
No, "heights" usually implies values. But if I scale down h, R increases.
Is there a constraint I missed?
"Interface: construct() -> sequence of nonnegative step heights".
Usually, these problems have a fixed L1 norm or L2 norm, or the "heights" are probabilities?
If the evaluator measures the raw value, and I can scale h, I should make h very small?
But "nonnegative heights" might imply integers? No, "floats" in the seed program.
Let's check the seed program's score: 1.1576.
If R can be arbitrarily large, the score would be huge.
Maybe the evaluator normalizes the output?
Or maybe "supremum norm" is defined differently?
Or maybe my inequality ||h*h||_inf >= ||h||_2^2 is wrong for some reason?
For non-negative h, (h*h)[k] = sum_j h_j h_{k-j}.
The term (h*h)[N-1] = sum_{j=0}^{N-1} h_j h_{N-1-j}.
This is NOT sum h_j^2 unless h is symmetric?
Ah! (h*h)[N-1] = sum h_j h_{N-1-j}.
This is the inner product of h and reversed h.
By Cauchy-Schwarz, |<h, rev(h)>| <= ||h||_2 ||rev(h)||_2 = ||h||_2^2.
So (h*h)[N-1] <= ||h||_2^2.
The maximum of the convolution might be LARGER than (h*h)[N-1].
For h = [1, 1], conv = [1, 2, 1]. Max = 2. ||h||_2^2 = 2. Ratio = 2/(2*2) = 0.5.
For h = [1, 0, 1], conv = [1, 0, 2, 0, 1]. Max = 2. ||h||_2^2 = 2. Ratio = 2/(2*2) = 0.5.
For h = [1, 0, 0, 1], conv = [1, 0, 0, 2, 0, 0, 1]. Max = 2. ||h||_2^2 = 2. Ratio = 0.5.
It seems for non-negative h, ||h*h||_inf is often close to ||h||_2^2 or larger.
Actually, for h >= 0, the max of conv is at the lag where the overlap is "best".
If h is "spread out", the peak is lower relative to L2?
Let's try h = [1, 1, 1, 1, 1] (N=5).
conv = [1, 2, 3, 4, 5, 4, 3, 2, 1]. Max = 5.
||h||_2^2 = 5. ||h||_1 = 5.
R = 5 / (5 * 5) = 1/5 = 0.2.
If I scale h by c, R(c h) = R(h) / c.
To maximize R, I need c to be small.
But the score is 1.15. This implies the evaluator might normalize h to have L1=1 or L2=1?
Or maybe the "heights" are constrained to sum to 1?
If sum h = 1, then R = ||h||_2^2 / ||h*h||_inf.
For uniform h (1/50), ||h||_2^2 = 1/50. ||h*h||_inf = 1/50. R = 1.
For delta h (1, 0...), ||h||_2^2 = 1. ||h*h||_inf = 1. R = 1.
For h = [0.5, 0.5, 0...], ||h||_2^2 = 0.5. conv = [0.25, 0.5, 0.25]. Max = 0.5. R = 0.5/0.5 = 1.
It seems R=1 is a lower bound? Or upper bound?
The current program scores 1.15. So R can be > 1.
This implies ||h*h||_inf < ||h||_2^2 is possible?
No, for non-negative h, (h*h)[k] >= 0.
Is it possible that the evaluator uses a different definition of autoconvolution?
"autoconvolution" usually means h * h.
Maybe the "supremum norm" is over a specific range?
Or maybe the heights can be negative? "nonnegative heights".
Let's look at the inspiration 1: "sparse support".
If support is sparse, the conv peak is small?
If h has disjoint supports, the conv peaks are separated.
But the max is still the sum of products.
Wait, if I use the spectral approach:
Minimize ||h*h||_inf for fixed ||h||_2.
This is equivalent to finding a "sidon set" like structure in the frequency domain?
Actually, if I represent h in Fourier domain, H(k).
||h*h||_inf is related to the max of the inverse FFT of |H|^2?
No, h*h in time domain is inverse FFT of H^2 (if using FFT convention).
Actually, (h*h)[n] = (1/N) sum_k H(k) H(k) e^{i 2 pi n k / N}?
If we use circular convolution, it's simpler.
But the problem says "on equal intervals of [-1/4, 1/4]". This sounds like linear convolution on a grid.
However, N=50 is small.
Let's assume the evaluator normalizes L1=1.
Then we want to maximize ||h||_2^2 / ||h*h||_inf.
This is equivalent to minimizing ||h*h||_inf / ||h||_2^2.
This ratio is the "peak-to-mean" power ratio of the autoconvolution?
To minimize the peak of the autoconvolution, we want the power spectrum |H(k)|^2 to be as flat as possible?
If |H(k)|^2 is constant, then h*h is a delta function (in circular conv).
For linear conv, it's more complex.
But "flat power spectrum" corresponds to a "noise-like" signal in time domain.
A random signal has a flat spectrum.
Let's try generating random signals and checking the score.
The current program uses parametric shapes.
I will use a spectral optimizer:
1. Initialize H(k) randomly (complex).
2. h = IFFT(|H|^2)? No, h must be non-negative.
   Better: h = |IFFT(H)|^2? Or h = IFFT(|H|^2) and clip?
   Actually, we can optimize H directly with constraints.
   
   Alternative:
   Use a "spectral shaping" filter.
   Let h = g * f, where g is a base shape and f is a filter?
   
   Let's stick to the directive: "superposition of non-negative, band-limited trigonometric polynomials".
   This implies h(t) = sum a_k cos(k t) + b_k sin(k t) + c0.
   We need h(t) >= 0.
   We want to minimize ||h*h||_inf.
   
   Let's try a different angle:
   The ratio R = ||h||_2^2 / (||h||_1 ||h*h||_inf).
   If we assume L1=1, R = ||h||_2^2 / ||h*h||_inf.
   We know ||h*h||_inf >= ||h||_2^2 is FALSE for linear conv?
   Let's re-verify.
   h = [1, -1] (if negative allowed). conv = [1, 0, -1]. Max = 1. ||h||_2^2 = 2. R = 2/(2*1) = 1.
   If h = [1, 1], conv = [1, 2, 1]. Max = 2. ||h||_2^2 = 2. R = 2/(2*2) = 0.5.
   It seems spreading mass reduces the ratio if L1 is fixed.
   Wait, if R can be > 1, then ||h*h||_inf < ||h||_2^2 must be possible.
   This requires negative values in h? But "nonnegative".
   OR, the "supremum norm" is defined differently.
   Maybe it's the L2 norm of the convolution? "supremum norm" is L_infinity.
   
   Let's look at the score 1.15 again.
   If the seed program gets 1.15, and it uses non-negative heights, then my analysis that R<=1 for non-negative h with L1=1 must be wrong.
   Where is the error?
   (h*h)[k] = sum_j h_j h_{k-j}.
   If h is non-negative, the max is at the lag with maximum overlap.
   For N=50, the max overlap is at lag 0 (if we consider circular) or center (if linear).
   For linear conv of length N, the peak is at N-1.
   (h*h)[N-1] = sum_{j=0}^{N-1} h_j h_{N-1-j}.
   This is <h, rev(h)>.
   By Cauchy-Schwarz, <h, rev(h)> <= ||h||_2 ||rev(h)||_2 = ||h||_2^2.
   So (h*h)[N-1] <= ||h||_2^2.
   However, the MAX of the convolution might be at a different lag!
   If h is not symmetric, the peak might be off-center?
   But for non-negative h, the sum is maximized when the large elements of h align with the large elements of rev(h).
   The maximum possible value of the convolution sum is indeed bounded by ||h||_2^2?
   No. Consider h = [1, 0, 0, 1].
   rev(h) = [1, 0, 0, 1].
   <h, rev(h)> = 1*1 + 0 + 0 + 1*1 = 2.
   ||h||_2^2 = 2.
   Conv: [1, 0, 0, 2, 0, 0, 1]. Max = 2.
   Ratio = 2 / (2 * 2) = 0.5 (if L1=2).
   If L1=1, h=[0.5, 0, 0, 0.5].
   ||h||_2^2 = 0.5.
   Conv: [0.25, 0, 0, 0.5, 0, 0, 0.25]. Max = 0.5.
   Ratio = 0.5 / (1 * 0.5) = 1.
   
   It seems R=1 is the maximum for non-negative h?
   Unless... the "autoconvolution" is normalized?
   Or maybe the "heights" are not constrained to sum to 1?
   If I can scale h, and R ~ 1/||h||_1, I can make R huge.
   But the score is 1.15.
   Maybe the evaluator fixes ||h||_2 = 1?
   If ||h||_2 = 1, R = 1 / (||h||_1 ||h*h||_inf).
   To maximize R, minimize ||h||_1 ||h*h||_inf.
   For h = delta, ||h||_1 = 1, ||h*h||_inf = 1. R = 1.
   For h = uniform, ||h||_1 = sqrt(N), ||h*h||_inf = 1/sqrt(N) (if ||h||_2=1, h_i=1/sqrt(N), conv max = N * (1/N) = 1? No.
   h_i = 1/sqrt(N).
   conv max = sum (1/sqrt(N))^2 = 1.
   ||h||_1 = sqrt(N).
   R = 1 / (sqrt(N) * 1) = 1/sqrt(N).
   So delta is better.
   
   This is confusing. Let's look at the inspiration 1: "sparse support".
   Sparse support usually means few non-zeros.
   If I use a "sidon set" of non-zeros, the autoconvolution peaks are distinct and small?
   If the non-zeros are far apart, the convolution peaks don't add up.
   This reduces ||h*h||_inf relative to ||h||_2^2?
   If h has support S, ||h||_2^2 = sum_{i in S} h_i^2.
   If S is a Sidon set, the convolution h*h has no large peaks except at 0?
   Actually, if S is sparse, the convolution is sparse.
   The peak at 0 is sum h_i^2.
   Other peaks are sums of products h_i h_j where i-j = k.
   If S is Sidon, each k has at most one pair (i,j).
   So other peaks are single products h_i h_j.
   By AM-GM, h_i h_j <= (h_i^2 + h_j^2)/2.
   So the peak at 0 is the largest?
   If so, ||h*h||_inf = ||h||_2^2.
   Then R = ||h||_2^2 / (||h||_1 ||h||_2^2) = 1/||h||_1.
   If I fix ||h||_1 = 1, R=1.
   If I don't fix ||h||_1, I can make R large.
   
   Maybe the task implies h is a probability vector (L1=1)?
   If so, R=1 is the max?
   But score is 1.15.
   Maybe the "supremum norm" is L2 norm of the convolution?
   "supremum norm" is L_inf.
   
   Let's assume the evaluator allows scaling and I should optimize the shape to maximize the "shape factor" and then scale it appropriately?
   But "score is value/seed". If I scale, I change the value.
   If I can make value arbitrarily large, the score would be infinite.
   So there MUST be a normalization in the evaluator.
   Most likely: The evaluator normalizes h to have L1=1 or L2=1.
   If L1=1, and R<=1, then 1.15 is impossible.
   If L2=1, and R<=1 (since ||h||_1 >= 1 and ||h*h||_inf >= 1 for non-negative? No, ||h*h||_inf can be < 1 if h is spread out?
   If h is uniform (L2=1), ||h*h||_inf = 1. ||h||_1 = sqrt(N). R = 1/sqrt(N) < 1.
   If h is delta (L2=1), ||h*h||_inf = 1. ||h||_1 = 1. R = 1.
   So R <= 1 for L2=1 too?
   
   Is it possible that h can be negative? "nonnegative heights".
   
   Let's try the spectral shaping strategy anyway. It's a structural change.
   I will optimize the Fourier coefficients to minimize the L_infinity norm of the autoconvolution, subject to non-negativity and fixed L2 norm (or L1).
   I will assume the evaluator normalizes L1=1.
   If R turns out to be > 1, great. If not, I'll be close to the seed.
   
   Implementation:
   1. Represent h as real part of IFFT of H.
   2. Enforce h >= 0 via penalty or projection.
   3. Minimize ||h*h||_inf.
   
   Actually, simpler:
   Use `scipy.optimize.minimize` on the Fourier coefficients (amplitudes and phases).
   Objective: -Ratio (maximize Ratio).
   Constraints: h >= 0.
   
   To ensure speed, I will use a coarse grid of Fourier modes (band-limited).
   """

import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We will search over a band-limited Fourier representation.
    # h[n] = sum_{k=0}^{K} A_k cos(2 pi k n / N) + B_k sin(2 pi k n / N) + C
    # To ensure non-negativity, we can use a "soft" constraint or projection.
    # Alternatively, we can parameterize h directly using a positive-definite kernel expansion?
    # No, let's stick to Fourier.
    
    # Let's try a simpler approach:
    # Randomly generate "Sidon-like" supports and optimize weights.
    # This was Inspiration 1.
    # The directive asks for "Spectral Shaping".
    
    # Spectral Shaping:
    # Minimize peak of conv(h,h) for fixed ||h||_2.
    # This is equivalent to minimizing the max of the inverse FFT of |H|^2 (circular approx).
    
    # Let's optimize the power spectrum P(k) = |H(k)|^2.
    # h = IFFT(sqrt(P) * phase).
    # To keep h real, H(-k) = conj(H(k)).
    # To keep h non-negative, we need careful phase selection.
    # A "random phase" spectrum often yields a Gaussian-like time domain signal (Central Limit Theorem).
    # Gaussian signals have low peak-to-average power ratio in time, but what about conv?
    
    # Let's try optimizing the amplitudes A_k of a cosine series:
    # h[n] = C + sum_{k=1}^{K} A_k cos(2 pi k n / N + phi_k)
    
    # We will use a random restart strategy with local optimization of amplitudes and phases.
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    K_max = 10  # Band limit
    
    def evaluate_fourier(params):
        # params: [C, A_1, phi_1, ..., A_K, phi_K]
        C = params[0]
        h = np.full(N, C)
        for k in range(1, K_max + 1):
            A = params[1 + 2*(k-1)]
            phi = params[2 + 2*(k-1)]
            h += A * np.cos(2 * np.pi * k * np.arange(N) / N + phi)
        
        # Enforce non-negativity by clipping (approximation)
        h = np.maximum(h, 0)
        
        # Normalize L1 to 1
        s = np.sum(h)
        if s < 1e-300:
            return 0.0
        h = h / s
        
        return compute_ratio(h), h

    # Random search over Fourier parameters
    num_trials = 2000
    for i in range(num_trials):
        if time.time() > deadline:
            break
            
        # Random parameters
        C = rng.uniform(0.5, 2.0)
        A = rng.uniform(0.0, 1.0, K_max)
        phi = rng.uniform(0, 2*np.pi, K_max)
        
        params = np.concatenate(([C], A, phi))
        
        score, h = evaluate_fourier(params)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # Local refinement if score is good
        if score > best_score * 0.95 and i % 10 == 0:
            # Use Nelder-Mead on a subset of params?
            # Too slow for 2000 trials.
            pass
            
    # Final refinement: Coordinate descent on the Fourier amplitudes of the best candidate?
    # We lost the params. Let's just return best_h.
    
    # One more strategy: "Flat Spectrum"
    # If we set A_k to be constant, we get a specific shape.
    # Let's try a few specific spectral shapes.
    
    # 1. White noise spectrum (random phases, constant amplitude)
    # 2. Low-pass spectrum (amplitude decays)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Random phases, constant amplitude (White Noise)
        C = 1.0
        A = np.ones(K_max) * 0.5
        phi = rng.uniform(0, 2*np.pi, K_max)
        params = np.concatenate(([C], A, phi))
        score, h = evaluate_fourier(params)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
        # Low-pass
        A = np.exp(-np.arange(1, K_max+1) * 0.5)
        phi = rng.uniform(0, 2*np.pi, K_max)
        params = np.concatenate(([C], A, phi))
        score, h = evaluate_fourier(params)
        if score > best_score:
            best_score = score
            best_h = h.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
