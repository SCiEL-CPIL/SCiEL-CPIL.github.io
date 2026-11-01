import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with a Blackman window, which has very low sidelobes
    # Blackman window: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
    n = np.arange(N)
    heights = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    
    # Normalize to unit energy to keep numbers stable
    heights = heights / np.linalg.norm(heights)
    
    best_heights = heights.copy()
    best_score = _compute_psl(heights)
    
    # Simulated Annealing
    rng = np.random.RandomState(42)
    T = 1e-4  # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline - 5.0:
        # Perturb a small subset of indices
        num_perturb = rng.randint(1, 5)
        indices = rng.choice(N, num_perturb, replace=False)
        
        new_heights = best_heights.copy()
        noise = rng.randn(num_perturb) * 0.01
        new_heights[indices] += noise
        
        # Re-normalize
        new_heights = new_heights / np.linalg.norm(new_heights)
        
        new_score = _compute_psl(new_heights)
        
        # Acceptance probability
        if new_score < best_score or rng.rand() < np.exp(-(new_score - best_score) / T):
            best_heights = new_heights
            best_score = new_score
        
        T *= cooling
    
    return best_heights

def _compute_psl(heights):
    """Compute the Peak Sidelobe Level."""
    # Autoconvolution
    conv = np.convolve(heights, heights, mode='full')
    # The peak is at the center (index N-1)
    center = N - 1
    main_lobe = conv[center]
    
    # If main lobe is zero (shouldn't happen with non-zero heights), return inf
    if abs(main_lobe) < 1e-10:
        return float('inf')
        
    # Sidelobes are everything else
    # We can mask out the center
    conv_masked = conv.copy()
    conv_masked[center] = 0.0
    
    max_sidelobe = np.max(np.abs(conv_masked))
    
    return max_sidelobe / abs(main_lobe)
# EVOLVE-BLOCK-END
