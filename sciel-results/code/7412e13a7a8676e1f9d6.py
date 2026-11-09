import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Optimizes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) 
    using Stochastic Local Search (Simulated Annealing).
    
    The objective function is scale-invariant, so we can fix ||h||_1 = 1 
    or just work with raw values and let the ratio normalize itself.
    However, working with normalized vectors avoids overflow/underflow issues 
    in the autoconvolution calculation if values get large.
    
    We start with a random or uniform initialization and perturb individual 
    elements, accepting changes that improve the score, and occasionally 
    accepting worse changes to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with uniform distribution (seed) or random
    # Uniform gives ratio = (N * (1/N)^2) / (1 * (N/N)) = 1/N = 0.02
    # Delta gives ratio = (1) / (1 * 1) = 1.0
    # We want to see if we can beat 1.0.
    
    # Let's start with a delta-like or random sparse distribution
    # to be closer to the likely optimum.
    h = np.zeros(N)
    # Random sparse initialization
    indices = np.random.choice(N, size=5, replace=False)
    h[indices] = 1.0
    h /= np.sum(h) # Normalize L1 to 1 for stability during search
    
    def compute_ratio(h_vec):
        # h_vec is non-negative
        l1 = np.sum(h_vec)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h_vec**2)
        
        # Autoconvolution: (h * h)[k] = sum_i h[i] h[i+k]
        # For discrete sequences, we can use np.convolve
        # Mode 'full' gives length 2N-1.
        # We need the supremum (max) of this convolution.
        # Since h is non-negative, the max will be at the center (lag 0) 
        # if the distribution is symmetric-ish, but generally we just take max.
        conv = np.convolve(h_vec, h_vec, mode='full')
        l_inf_conv = np.max(conv)
        
        if l_inf_conv == 0:
            return 0.0
            
        return (l2_sq) / (l1 * l_inf_conv)

    best_h = h.copy()
    best_score = compute_ratio(best_h)
    
    # Simulated Annealing parameters
    T = 1e-5  # Initial temperature
    cooling_rate = 0.999
    iterations = 0
    
    # We will perform local moves: pick an index, change its value, renormalize.
    # Or simply: pick index i, add delta to h[i], subtract from h[j] to keep L1 constant?
    # Actually, since the ratio is scale invariant, we don't need to keep L1 constant.
    # We can just modify h[i] directly.
    
    # To ensure non-negativity, we use multiplicative updates or clamped additive.
    # Let's try: h[i] *= (1 + noise)
    
    while time.time() < deadline:
        # Pick a random index to perturb
        i = np.random.randint(N)
        
        # Generate a multiplicative factor
        # Log-normal distribution for multiplicative noise
        factor = np.exp(np.random.normal(0, 0.1))
        
        h_new = h.copy()
        h_new[i] *= factor
        
        # Ensure non-negativity (already guaranteed by exp)
        # Optional: Normalize to prevent drift? 
        # The ratio is scale invariant, so drift doesn't matter for the score, 
        # but numerical stability might suffer if values become huge/tiny.
        # Let's renormalize to L1=1 periodically or always.
        sum_h = np.sum(h_new)
        if sum_h > 0:
            h_new = h_new / sum_h
        
        score_new = compute_ratio(h_new)
        
        if score_new > best_score:
            best_score = score_new
            best_h = h_new
            h = h_new
        else:
            # Accept worse move with probability exp((score_new - score)/T)
            # Note: score_new < score
            delta = score_new - score
            if np.random.rand() < np.exp(delta / T):
                h = h_new
            # else: reject, keep h as is
        
        T *= cooling_rate
        if T < 1e-10:
            T = 1e-10
        iterations += 1
        
        # Check time more frequently
        if iterations % 100 == 0 and time.time() > deadline:
            break

    # Return the best sequence found as a list of floats
    # The evaluator expects a sequence of nonnegative step heights.
    # We can return best_h directly. Scaling doesn't change the ratio, 
    # but returning the normalized version is clean.
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
