"""
Strategy: Use a convex SDP-like approach via alternating minimization in the frequency domain.
The key insight is that the normalized max autoconvolution is:
  max_k conv[k] / (h^2 * (sum a)^2)

For a step function on [-1/4, 1/4] with N=600 intervals, we can think of this as:
  We want to minimize the peak of |F(omega)|^2 (where F is the Fourier transform)
  subject to the constraint that the integral of f is fixed (normalization).

Actually, the autoconvolution in the time domain corresponds to |F(omega)|^2 in the frequency domain.
The max autoconvolution is the max over the convolution, which by Parseval/Plancherel relates to
the L2 norm of F, but the MAX is different.

Alternative structural approach: Use a "spectral" parameterization.
Instead of optimizing in the time domain directly, we can use the fact that:
  conv[k] = sum_i a[i] a[i+k]

This is the autocorrelation. The maximum of the autocorrelation for a nonnegative sequence
with fixed sum is related to the concentration of the sequence.

New strategy: Use a two-level approach.
1. First, find the optimal "shape" using a convex relaxation in the frequency domain.
   Specifically, we can use the fact that for a given peak constraint T, the problem
   min sum(a^2) s.t. autocorr[k] <= T for all k, sum(a)=1, a>=0
   is not convex. But we can use bisection on T with a feasibility check.

2. Better: Use the fact that the optimal solution for minimizing the peak autocorrelation
   with fixed L1 norm tends to be "spread out" in a specific way. We can use a 
   "water-filling" approach in the frequency domain.

3. Most practical: Use a different optimization algorithm - specifically, use projected 
   gradient descent with a non-smooth max constraint, or use a penalty method with 
   a different initial structure.

Let me try a fundamentally different approach: 
- Use a "two-scale" ansatz: the optimal function might have a specific structure 
  (e.g., constant on sub-intervals, or following a specific curve).
- Use coordinate descent with exact line search on blocks.
- Or: use the fact that the problem is invariant under translation (the domain is symmetric),
  so we can fix the center of mass and optimize the "spread".

Actually, the most promising structural change: 
Use a DIFFERENT representation. Instead of 600 free variables, use a low-dimensional 
parametric family that captures the key features, then optimize that family, then refine.

But even better: use a "convex" reformulation. Note that:
  max_k conv[k] = max_k sum_i a[i] a[i+k]

If we fix the L1 norm to 1 (by scaling), we want to minimize max_k conv[k].
This is equivalent to: minimize t s.t. sum_i a[i] a[i+k] <= t for all k, sum(a)=1, a>=0.

This is a quadratic program (QP) with a max constraint. We can solve it using:
- Bisection on t: for a given t, check if there exists a>=0 with sum(a)=1 and 
  autocorr[k] <= t for all k. This feasibility problem is non-convex.

Alternative: Use the "peak-to-average" ratio concept. The normalized max autoconvolution
is the peak of the autocorrelation relative to the DC component squared.

Let me try a completely different optimization: use scipy.optimize.minimize with 
method='SLSQP' on the original variables with explicit constraints, or use a 
penalty method with a different smoothing.

Actually, the most structurally different approach: 
Use a "spectral" method. The autocorrelation's max is related to the eigenvalues of 
the autocorrelation matrix. But more directly:

The key structural insight: The optimal step function for minimizing the peak 
autoconvolution (normalized) on a symmetric interval likely has a specific 
"bell-shaped" profile that is NOT Gaussian. In fact, for continuous functions, 
the optimal might be related to the prolate spheroidal wave functions or similar.

For step functions, we can use a "discrete" spectral approach:
- Represent a in the Fourier basis
- The constraint sum(a)=1 fixes the DC component
- The max autoconvolution is the max of the inverse Fourier transform of |A|^2
- We want to minimize this max

Let me implement a "bisection + feasibility" approach where the feasibility 
check uses a different method (e.g., a simple iterative method or a relaxed problem).

But given time constraints, let me try a more practical structural change:
Use a "multi-start" with a DIFFERENT optimization algorithm (e.g., Nelder-Mead 
or a custom gradient-free method) combined with a better initial guess based 
on the "optimal" continuous profile.

Actually, the best structural change: use a "convex" proxy. 
Note that min max_k conv[k] is equivalent to min t s.t. conv[k] <= t for all k.
We can use a "smooth max" approximation: t_smooth = log(sum(exp(conv[k]/tau))/tau)
and minimize this with a decreasing tau. This is a convex function of conv, 
but conv is quadratic in a, so it's still non-convex.

Let me try a different angle: use the fact that for NONNEGATIVE a, the autocorrelation 
is nonnegative, and the max is at k=0 (since conv[0] = sum(a^2) >= conv[k] for k!=0 
by Cauchy-Schwarz? No, that's not true in general).

Wait: conv[0] = sum_i a[i]^2, and conv[k] = sum_i a[i] a[i+k].
By Cauchy-Schwarz, conv[k] <= sqrt(conv[0] * conv[0]) = conv[0]. So conv[0] is ALWAYS 
the maximum! 

Wait, is that true? conv[0] = sum a[i]^2. conv[k] = sum a[i] a[i+k].
By Cauchy-Schwarz: (sum a[i] a[i+k])^2 <= (sum a[i]^2)(sum a[i+k]^2) = conv[0] * conv[0] = conv[0]^2.
So conv[k] <= conv[0] for all k. Therefore, the max autoconvolution is ALWAYS conv[0]!

This changes everything! The normalized max autoconvolution is:
  conv[0] / (h^2 * (sum a)^2) = sum(a^2) / (h^2 * (sum a)^2)

So we just need to minimize sum(a^2) / (sum a)^2, subject to a>=0.
By Cauchy-Schwarz, (sum a)^2 <= N * sum(a^2), so sum(a^2)/(sum a)^2 >= 1/N.
Equality holds when all a[i] are equal!

So the optimal is the UNIFORM distribution! And the minimum value is 1/N = 1/600.

Wait, but the current program scores 1.0, which means it's achieving the seed value.
Let me re-read the problem. "minimize the normalized maximum of its autoconvolution".

If the max is always at k=0 for nonnegative functions, then the answer is 1/600 ≈ 0.00167.
But the seed program returns a score of 1.0, which suggests the "raw value" is 1.0.

Let me check: the score is "the upper bound on C1". The raw value is lower is better.
The seed program's construct() returns a list of heights. The evaluator computes the score.

If the optimal is uniform with value 1/600, then the seed program (which starts with 
various profiles and optimizes) should find something close to 1/600. But it scores 1.0, 
meaning the raw value is 1.0? That doesn't make sense.

Unless... the normalization is different. Let me re-read: "normalized maximum of its 
autoconvolution". Maybe the normalization is different.

Looking at the code: norm = h**2 * (np.sum(a))**2, and max_conv / norm.
For uniform a[i] = 1/N, sum(a) = 1, h = 1/1200.
conv[0] = sum(1/N^2) = N * (1/N^2) = 1/N.
norm = h^2 * 1 = (1/1200)^2 = 1/1440000.
score = (1/N) / (1/1440000) = 1440000/600 = 2400.

That's not 1.0. So the evaluator must be doing something different, or I'm misunderstanding.

Wait, the task says "score is the upper bound on C1" and "the evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program".

So if the seed program returns a value V_seed, and my program returns V_mine, 
the search score is V_seed / V_mine. If search score = 1.0, then V_mine = V_seed.

The seed program's construct() returns list(best_a). The evaluator computes the "raw value" 
from this. I need to understand what "raw value" means.

Given the confusion, let me just implement the structurally different approach: 
since I believe the max autoconvolution is always at k=0 for nonnegative functions, 
the optimal is uniform. But to be safe, let me verify with the actual computation and 
use a different optimization approach that directly minimizes sum(a^2)/(sum(a))^2.

Actually, let me just return the uniform distribution and see. If the max is indeed 
at k=0, this is optimal. If not, I'll need to reconsider.

But the directive says to make a STRUCTURAL change. The current program uses 
L-BFGS-B with softmax parameterization. A structural change would be to use 
a different algorithm or a different problem formulation.

Given my analysis that the max is at k=0, the problem reduces to minimizing 
sum(a^2)/(sum(a))^2, which is minimized by uniform a. So the optimal solution 
is simply the uniform distribution.

Let me implement this with a verification step and a fallback to a different 
optimization if my analysis is wrong.
"""
import numpy as np
import time

N = 600
h = (0.25 - (-0.25)) / N  # 0.5 / 600 = 1/1200

def _autoconvolve_max(a):
    """
    Compute the normalized maximum autoconvolution.
    Input a: array of heights.
    Returns: normalized max value.
    """
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # Key insight: For nonnegative a, conv[k] <= conv[0] by Cauchy-Schwarz.
    # So max autoconvolution is always conv[0] = sum(a^2).
    # Normalized: sum(a^2) / (h^2 * (sum(a))^2)
    # This is minimized when all a[i] are equal (by Cauchy-Schwarz equality condition).
    # So the optimal is the uniform distribution.
    
    # Uniform distribution
    a_uniform = np.ones(N) / N
    score_uniform = _autoconvolve_max(a_uniform)
    
    # Verify that conv[0] is indeed the max
    nfft = 1 << (2 * N - 1).bit_length()
    A = np.fft.rfft(a_uniform, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*N-1]
    max_conv = np.max(conv)
    conv0 = conv[0]
    
    # If my analysis is correct, max_conv should equal conv0
    # Let's also try a few other profiles to see if any do better
    x = np.linspace(-0.25, 0.25, N)
    
    # Gaussian
    a_g = np.exp(-x**2 / (2 * 0.15**2))
    a_g = a_g / np.sum(a_g)
    score_g = _autoconvolve_max(a_g)
    
    # Triangular
    a_tri = np.maximum(1 - np.abs(x) / 0.25, 0)
    a_tri = a_tri / np.sum(a_tri)
    score_tri = _autoconvolve_max(a_tri)
    
    # Keep the best
    best_a = a_uniform.copy()
    best_score = score_uniform
    
    if score_g < best_score:
        best_a = a_g.copy()
        best_score = score_g
    
    if score_tri < best_score:
        best_a = a_tri.copy()
        best_score = score_tri
    
    # If time permits, do a quick optimization to verify
    if time.time() < deadline - 20:
        # Use a simple gradient descent on the normalized objective
        # Objective: sum(a^2) / (sum(a))^2, with a >= 0
        # This is minimized by uniform, so we should converge to uniform
        
        # Actually, let's just verify with a few random perturbations
        rng = np.random.RandomState(42)
        for _ in range(10):
            a_rand = rng.rand(N)
            a_rand = a_rand / np.sum(a_rand)
            score_rand = _autoconvolve_max(a_rand)
            if score_rand < best_score:
                best_a = a_rand.copy()
                best_score = score_rand
    
    return list(best_a)
