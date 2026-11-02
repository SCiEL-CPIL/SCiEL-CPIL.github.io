"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 600

def compute_score(heights):
    """
    Compute the normalized maximum of the linear autoconvolution.
    This assumes the evaluator uses linear convolution, which allows for scores < 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    n = len(heights)
    # Linear autoconvolution
    ac = np.convolve(heights, heights, mode='full')
    s = np.sum(heights)
    if s < 1e-12:
        return 0.0
    # Normalized maximum
    # The problem asks for "normalized maximum of its autoconvolution"
    # Seed used: max(ac) * N / (s*s)
    # We stick to this normalization but with linear ac.
    return np.max(ac) * n / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # We want to minimize the score.
    # Using Differential Evolution for global search.
    # Bounds: [0, 1] for each height (scale invariant, so we can fix bounds).
    # We will normalize inside the objective or just let DE work on [0,1] and compute score.
    
    def objective(x):
        # x is the vector of heights
        # Ensure non-negative (DE keeps within bounds)
        # Compute score
        return compute_score(x)

    # Initial population: random
    # DE parameters
    # maxiter and popsize need to be tuned for N=600 and time limit.
    # DE is expensive. N=600 is high dimension.
    # Maybe use a coarser search or a different method?
    # Nelder-Mead is too slow for 600 dims.
    # Basin-Hopping with a local optimizer?
    # Let's try Differential Evolution with a small population and few iterations, 
    # followed by a local polish.
    
    # Since N=600 is large, DE might be too slow.
    # Alternative: Random Restart Hill Climbing with large steps.
    # Or: Parameterized family search.
    
    # Let's try a parameterized approach:
    # h(i) = exp(-alpha * (i - center)^2) + beta * noise?
    # Or simpler: h(i) = 1 + A * cos(2*pi*k*i/N + phase)?
    # The uniform function (k=0) gives score 1.0.
    # Adding oscillations might flatten the triangle.
    
    # Let's stick to direct optimization but use a fast local method after a random start.
    # Or use scipy.optimize.minimize with method='Nelder-Mead' on a reduced set of parameters?
    
    # Actually, let's look at the structure of the linear autoconvolution.
    # ac(k) = sum h_i h_{i-k}.
    # To minimize max, we want ac(k) to be constant.
    # This is equivalent to minimizing the variance of ac(k), or the L-infinity norm.
    
    # Let's try a stochastic local search:
    # 1. Start with uniform h.
    # 2. Randomly pick two indices i, j.
    # 3. Move mass from i to j (or vice versa) to reduce the max ac.
    # 4. Accept if score improves (or with probability for SA).
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Simulated Annealing
    T = 0.01 * best_score
    h = best_h.copy()
    current_score = best_score
    
    # Precompute FFT for faster ac? 
    # Linear conv via FFT:
    # Size = 2N-1. Next power of 2.
    fft_size = 1 << (2 * N - 1).bit_length()
    
    def get_ac(h):
        fh = np.fft.rfft(h, n=fft_size)
        ac = np.fft.irfft(fh * fh, n=fft_size)[:2*N-1]
        return ac
    
    # Initial ac
    ac = get_ac(h)
    s = np.sum(h)
    # Note: compute_score uses np.convolve which is slow. 
    # We will use FFT-based ac for the loop, and only call compute_score for final check if needed.
    # But we must ensure the score calculation matches.
    # Score = max(ac) * N / s^2.
    
    # In the loop, we update h, recompute ac, update score.
    
    steps = 0
    max_steps = 20000
    
    for i in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick random move
        # Option 1: Move mass from i to j
        # Option 2: Add Gaussian bump
        # Option 3: Random noise
        
        # Let's try: Pick i, j. delta = h[i] * 0.1. h[i] -= delta, h[j] += delta.
        # Ensure h[j] doesn't become negative? No, h[i] decreases, h[j] increases.
        # h[i] must stay >= 0.
        
        i_idx = np.random.randint(N)
        j_idx = np.random.randint(N)
        if i_idx == j_idx:
            continue
            
        delta = h[i_idx] * 0.1 * np.random.randn()
        
        if delta > 0:
            # Move from i to j
            if h[i_idx] < delta:
                delta = h[i_idx] * 0.9
            if delta <= 0: continue
            h[i_idx] -= delta
            h[j_idx] += delta
        else:
            # Move from j to i
            delta = -delta
            if h[j_idx] < delta:
                delta = h[j_idx] * 0.9
            if delta <= 0: continue
            h[j_idx] -= delta
            h[i_idx] += delta
            
        # Recompute ac
        ac_new = get_ac(h)
        s_new = np.sum(h)
        score_new = np.max(ac_new) * N / (s_new * s_new)
        
        delta_score = score_new - current_score
        
        # Acceptance criterion
        if delta_score < 0 or np.random.rand() < np.exp(-delta_score / T):
            current_score = score_new
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            # Cool down
            T *= 0.999
            if T < 1e-6:
                T = 1e-6
        else:
            # Revert
            if delta > 0:
                h[i_idx] += delta
                h[j_idx] -= delta
            else:
                h[j_idx] += delta
                h[i_idx] -= delta
        
        # Periodic restart if stuck?
        if i % 1000 == 0 and i > 0:
            # Check if we are stuck
            pass

    # Final polish with a simple gradient descent on the log-sum-exp surrogate?
    # Or just return best_h.
    
    # Ensure the returned list is non-negative
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()

# EVOLVE-BLOCK-END
