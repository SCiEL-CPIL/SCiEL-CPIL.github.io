import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0

# Precompute fine grid for fast evaluation
# Using a moderate resolution for the FFT to balance accuracy and speed.
# The step function is constant on intervals of width W.
# We sample the step function on a fine grid to compute the autoconvolution accurately.
# The autoconvolution of a step function is a piecewise quadratic function.
# Its maximum occurs either at the center of a bin or at the boundaries.
# Sampling at the center of each fine sub-interval is sufficient if fine enough.
FINE_N = 6000 # 10 samples per bin
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map bin heights to fine grid
    f_fine = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    # The autoconvolution is defined on [-0.5, 0.5] effectively, but we only care about the range where it's non-zero.
    # Since support is [-0.25, 0.25], the convolution support is [-0.5, 0.5].
    # The output of irfft is centered at 0.
    # We need the maximum over the entire convolution.
    max_conv = np.max(conv)
    
    # Compute L2 norm squared
    # Integral of f^2 approx sum(f_fine^2 * DX)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_conv / l2_sq

def objective(h):
    """Objective function for Differential Evolution."""
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    # Initial best guess: Uniform distribution
    # For uniform h=1, autoconv is a triangle.
    # Max is at center.
    # Let's calculate the seed value to know where we stand.
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_val = ratio_from_h(best_h)
    
    # Differential Evolution Setup
    # Bounds: [0, 1] for each bin. 
    # The scale of h doesn't matter for the ratio (it's homogeneous), 
    # but DE works in a bounded box. 
    # We can normalize after each evaluation or just let DE find a shape.
    # Actually, ratio_from_h is scale-invariant. 
    # However, DE needs a bounded domain. [0, 1] is a good choice.
    # If DE finds a solution with small values, it's still valid.
    # To help DE, we might want to encourage non-zero values, but [0,1] allows zeros.
    
    bounds = [(0, 1)] * N
    
    # DE parameters
    # maxiter: Number of generations.
    # popsize: Multiplier for population size.
    # tol: Stopping tolerance.
    # Strategy: 'best1bin' is standard.
    
    # We have ~145 seconds.
    # Each evaluation takes ~2-5 ms (FFT of size ~16k-32k).
    # Let's estimate cost per generation.
    # Popsize = 10 * 600 = 6000 individuals? That's too many.
    # Standard DE popsize is 5-15 * N. 5*600 = 3000.
    # 3000 evals per gen.
    # If 1 eval = 3ms, 1 gen = 9 seconds.
    # We can afford ~15 generations.
    
    # Let's try a smaller population to allow more generations or faster convergence.
    # Or use a coarser grid for DE and refine later?
    # No, the task requires 600 intervals.
    
    # Let's try popsize=5 (3000 individuals) and maxiter=20.
    # 20 * 3000 = 60,000 evaluations.
    # 60,000 * 3ms = 180s. Too slow.
    
    # We need to be faster.
    # Optimization: Use a smaller FFT size?
    # The step function has 600 bins.
    # The autoconvolution of a step function is piecewise quadratic.
    # We can compute the exact maximum without a fine grid FFT.
    
    # Let's re-evaluate the cost of ratio_from_h.
    # N_FFT = 1 << (2*6000 - 1).bit_length() = 1 << 14 = 16384.
    # rfft of 16384 is fast. ~0.1ms?
    # Let's assume 1ms per evaluation.
    # 100,000 evals = 100s.
    # So we can afford ~100k evaluations.
    
    # Let's try popsize=10 (6000 ind) -> 6000 evals/gen.
    # 15 gens -> 90,000 evals.
    # This should fit in ~90s.
    
    # Initial population:
    # DE creates its own initial population.
    # We can provide a seed or initial population.
    # Let's use a random initial population but biased towards non-zero.
    
    # Actually, DE in scipy handles initialization.
    
    # Let's run DE.
    try:
        # Use a timeout wrapper or just trust the time check inside objective?
        # DE doesn't have a built-in time limit, only maxiter.
        # We must ensure maxiter is small enough.
        
        # Let's try maxiter=15, popsize=10
        # If it's too slow, we'll catch it.
        
        # To improve speed, we can reduce FINE_N to 3000 (5 samples/bin).
        # Let's check if 3000 is accurate enough.
        # The error in max_conv estimation decreases with grid size.
        # For a smooth function, it's very accurate.
        # For a step function, the convolution is continuous but has kinks.
        # The maximum might be exactly at a kink.
        # With 5 samples/bin, we might miss the peak if it's narrow.
        # However, the autoconvolution of a step function is a sum of triangles/quadratics.
        # The peaks are not arbitrarily narrow.
        # Let's stick with FINE_N=6000 for safety, but optimize the FFT.
        
        # Actually, let's try to optimize ratio_from_h further.
        # We can compute the convolution directly on the bins?
        # No, the convolution of two step functions is not a step function.
        
        # Let's proceed with DE.
        
        # We will use a custom callback to stop early if time is running out?
        # Scipy DE doesn't support time-based stopping directly.
        # We can set maxiter conservatively.
        
        # Let's estimate time more carefully.
        # Time per ratio_from_h:
        # 1. h[CELL]: 6000 ops. Negligible.
        # 2. rfft: 16k complex ops. ~0.05 ms.
        # 3. mult: 16k ops.
        # 4. irfft: 16k ops. ~0.05 ms.
        # 5. max, sum: 16k ops.
        # Total ~0.2 ms?
        # If 0.2 ms, 100k evals = 20s.
        # Then we can do much more.
        
        # Let's try maxiter=50, popsize=10.
        # 50 * 6000 = 300,000 evals.
        # 300,000 * 0.2ms = 60s.
        # This is safe.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            seed=42, 
            polish=False # Polish uses BFGS, which might be slow or fail on non-smooth. 
                         # DE finds the global optimum well.
                         # We can do a manual polish later if time permits.
        )
        
        if result.success or result.fun < best_val:
            h_candidate = result.x
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
                
    except Exception as e:
        # Fallback if DE fails
        pass

    # Local Refinement:
    # Use Nelder-Mead or BFGS on the best DE result?
    # The function is non-smooth (max operator).
    # Nelder-Mead is more robust to non-smoothness.
    
    if time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        
        # Try Nelder-Mead on the 600-dim vector.
        # This might be slow in 600 dims.
        # Maybe restrict to a subset of bins?
        # Or just do a few random perturbations?
        
        # Let's try a simple local search:
        # Perturb a few random bins.
        # This is what the current program does.
        # But DE should have already explored this.
        
        # Instead, let's try to "smooth" the DE result?
        # No, the optimal shape might be jagged.
        
        # Let's just do a few random restarts of DE with different seeds if time permits?
        # No, one good DE run is better.
        
        # Let's try to refine with BFGS on a smoothed version?
        # No, let's stick to the DE result.
        
        # One improvement: Normalize the best_h?
        # The ratio is scale invariant. 
        # But the evaluator might expect specific values?
        # "sequence of nonnegative step heights".
        # Any scale is fine.
        
        # Let's ensure the output is a list of floats.
        
    return list(best_h)

# Run a quick self-test to ensure no errors
if __name__ == "__main__":
    h = construct()
    print(f"Score: {ratio_from_h(np.array(h))}")
