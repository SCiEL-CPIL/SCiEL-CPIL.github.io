"""
Strategy: Continuous spectral optimization followed by discrete refinement.
The original approach used a naive O(N^2) or O(N log N * N) local search which is too slow for 400 steps in Python loops.
This approach:
1. Uses L-BFGS-B to optimize a continuous relaxation of the problem (minimizing sum of squared autocorrelation sidelobes).
   This is a standard technique for finding low-peak sidelobe sequences (like Barker codes) by navigating the non-convex landscape.
2. Maps the continuous solution to a binary sequence {-1, 1}.
3. Performs a fast, vectorized discrete local search (bit-flips) to refine the binary sequence.
   The discrete search is optimized to compute the change in the objective efficiently.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _compute_autocorr(x):
    """Compute the full autocorrelation of x using FFT."""
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac

def _max_sidelobe(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    ac = _compute_autocorr(x)
    return np.max(np.abs(ac[1:]))

def _objective_and_gradient(z):
    """
    Objective: Minimize sum of squares of autocorrelation sidelobes.
    z is a vector of length N.
    We use the fact that autocorrelation is the inverse FFT of |FFT(z)|^2.
    To get the gradient, we can use the chain rule or automatic differentiation logic.
    However, for L-BFGS-B, providing an analytical gradient is much faster.
    
    Let A(k) = sum_{j=0}^{N-1-k} z_j z_{j+k} for k=1..N-1.
    We want to minimize S = sum_{k=1}^{N-1} A(k)^2.
    
    Gradient dS/dz_i:
    A(k) involves z_i if j=i (term z_i z_{i+k}) or j+k=i (term z_{i-k} z_i).
    dA(k)/dz_i = z_{i+k} (if i+k < N) + z_{i-k} (if i-k >= 0).
    Note: A(k) is defined for linear autocorrelation.
    
    Let's define the gradient vector g.
    g_i = 2 * sum_{k=1}^{N-1} A(k) * dA(k)/dz_i
         = 2 * sum_{k=1}^{N-1} A(k) * ( z_{i+k} * 1_{i+k < N} + z_{i-k} * 1_{i-k >= 0} )
    
    This can be computed efficiently using convolution or FFT, but for N=400, even a direct loop might be okay if we vectorize the inner sum?
    Actually, let's just use numerical gradient or a simpler approach:
    Since N=400 is small, we can compute the gradient by looping over k and updating g.
    Or better, use the property that the gradient is related to the convolution of A with z.
    
    Let's stick to a robust numerical gradient if analytical is tricky, but L-BFGS-B needs it.
    Actually, let's try a simpler objective: just minimize the max sidelobe? No, non-differentiable.
    Sum of squares is good.
    
    Let's implement the analytical gradient carefully.
    """
    z = np.array(z, dtype=float)
    ac = _compute_autocorr(z)
    # Sidelobes are ac[1:N]
    side = ac[1:N]
    obj = np.sum(side ** 2)
    
    # Gradient computation
    # g_i = 2 * sum_{k=1}^{N-1} side[k-1] * (z[i+k] if i+k<N else 0 + z[i-k] if i-k>=0 else 0)
    # Let's construct this vector.
    # term1: z[i+k] for k=1..N-1-i. This is a correlation of side with z.
    # term2: z[i-k] for k=1..i. This is also a correlation.
    
    # Using np.correlate?
    # side is length N-1. z is length N.
    # We want for each i, sum_{k} side[k] * z[i+k] and sum_{k} side[k] * z[i-k].
    
    # Let's use FFT for speed and accuracy.
    # The gradient of sum A(k)^2 w.r.t z is 2 * (A * z) in some shifted sense.
    # Specifically, if A = autocorr(z), then grad = 2 * convolve(A_padded, z) ?
    
    # Let's just do the O(N^2) loop for the gradient since N=400 is small enough for L-BFGS iterations?
    # L-BFGS might take ~100-1000 iterations. 400^2 = 160,000 ops per gradient. 1000 * 160k = 160M ops.
    # In Python, this might be slow.
    # Let's use vectorized numpy operations.
    
    # g_i = 2 * (sum_{k=1}^{N-1-i} side[k-1] * z[i+k] + sum_{k=1}^{i} side[k-1] * z[i-k])
    
    # Create a matrix M where M[i, k] = 1 if z[i+k] is involved, etc.
    # Or use np.convolve.
    
    # Let's try a direct vectorized approach:
    # side is index 0..N-2 corresponding to lag 1..N-1.
    
    # Part 1: z[i+k]
    # For a fixed k, we add side[k-1] * z[k:N] to g[0:N-k]
    # Part 2: z[i-k]
    # For a fixed k, we add side[k-1] * z[0:N-k] to g[k:N]
    
    grad = np.zeros(N)
    # Vectorized accumulation
    # This loop is over N-1 (399) iterations. Each does a vector add of length ~N.
    # 399 * 400 ~ 160,000 ops. Very fast in numpy.
    for k in range(1, N):
        val = side[k-1]
        # Contribution from z[i+k] term: affects g[i] for i=0..N-1-k
        grad[:N-k] += val * z[k:]
        # Contribution from z[i-k] term: affects g[i] for i=k..N-1
        grad[k:] += val * z[:N-k]
        
    grad *= 2.0
    
    return obj, grad

def _discrete_refinement(x, time_limit):
    """
    Fast discrete local search.
    x is binary {-1, 1}.
    We try to flip bits to reduce max |ac[1:N]|.
    To be fast, we don't recompute full FFT every time.
    We can approximate the change or just do a few passes of full recomputation if N is small?
    N=400. FFT is ~10-20 us. 400 flips * 400 indices = 160,000 FFTs? No, 1 FFT per flip check.
    If we check all 400 bits in one pass, that's 400 FFTs. 400 * 20us = 8ms per pass.
    We can do ~1000 passes in 8 seconds. This is feasible.
    """
    start_time = time.time()
    best_x = x.copy()
    best_score = _max_sidelobe(best_x)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try flipping each bit
        # We can vectorize the "potential" check? No, max is non-linear.
        # Just loop.
        for i in range(N):
            new_x = best_x.copy()
            new_x[i] = -new_x[i]
            new_score = _max_sidelobe(new_x)
            if new_score < best_score - 1e-9:
                best_x = new_x
                best_score = new_score
                improved = True
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Continuous Optimization
    # Initialize with random continuous values
    rng = np.random.default_rng(42)
    x0 = rng.uniform(-1, 1, N)
    
    # Run L-BFGS-B
    try:
        res = minimize(_objective_and_gradient, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        x_cont = res.x
    except Exception:
        x_cont = x0
    
    # 2. Discretize
    # Threshold at 0
    x_bin = np.sign(x_cont)
    x_bin[x_bin == 0] = 1
    
    # 3. Discrete Refinement
    # Allocate remaining time
    elapsed = time.time() - start_time
    time_left = max(0, 130 - elapsed)  # Leave 10s buffer
    
    if time_left > 0:
        x_bin, score = _discrete_refinement(x_bin, time_left)
    else:
        score = _max_sidelobe(x_bin)
        
    return x_bin

# EVOLVE-BLOCK-END
