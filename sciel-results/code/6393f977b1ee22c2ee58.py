import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial guess: Random walk or random normal?
    # The inspiration mentions "Random walk with negative drift".
    # Let's start with a random normal distribution and optimize.
    # We want to minimize value = abs(2 * N * max(conv(h, h)) / sum(h)**2)
    
    # We can scale h freely, so we fix sum(h) = 1 to simplify?
    # No, the metric is scale invariant.
    # Let's fix sum(h) = N (like the seed) and minimize max(conv).
    # Or fix max(conv) and maximize sum?
    
    # Let's use a gradient-free optimization (e.g., Nelder-Mead or simple random perturbations)
    # But N=400 is high dimensional.
    
    # Alternative: Use the property that we want h to be "flat" in convolution.
    # This is related to "Constant Amplitude Zero Autocorrelation" (CAZAC) sequences,
    # but we want low PEAK, not just zero sidelobes.
    # Actually, we want the convolution to be as flat as possible.
    
    # Strategy:
    # 1. Start with h = ones(N)
    # 2. Perturb h randomly
    # 3. Calculate value
    # 4. If better, keep.
    
    # To speed up, we can use FFT for convolution.
    # np.convolve is O(N^2). FFT is O(N log N).
    
    def compute_value(h):
        s = np.sum(h)
        if np.isclose(s, 0):
            return np.inf
        # Use FFT for full convolution
        # Pad to length 2N-1
        size = 2 * N - 1
        f_h = np.fft.fft(h, size)
        conv = np.fft.ifft(f_h * f_h).real
        # The convolution result might have small imaginary parts due to precision
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    # Initialize
    h = np.ones(N)
    best_h = h.copy()
    best_val = compute_value(h)
    
    # Optimization loop
    # Use a random restart + local search approach
    
    current_h = h.copy()
    current_val = best_val
    
    step_size = 1.0
    
    while time.time() < deadline:
        # Generate a perturbation
        # Try different noise distributions
        if np.random.rand() < 0.5:
            noise = np.random.randn(N) * step_size
        else:
            noise = np.random.randint(-1, 2, size=N) * step_size
        
        candidate_h = current_h + noise
        
        # Optional: Enforce zero mean of the AC component?
        # No, we want large DC.
        
        cand_val = compute_value(candidate_h)
        
        if cand_val < current_val:
            current_h = candidate_h
            current_val = cand_val
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
            # Reduce step size on success
            step_size *= 0.95
        else:
            # Increase step size on failure (simulated annealing style)
            step_size *= 1.05
            if step_size > 10.0:
                step_size = 10.0
    
    return best_h
# EVOLVE-BLOCK-END
