import numpy as np

N = 400

def construct():
    # Start with seed
    best_h = np.ones(N)
    best_val = np.inf
    
    # Replicate evaluator
    def get_val(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))
    
    current_val = get_val(best_h)
    
    # Random search
    # Use a fixed seed for reproducibility in the solution code?
    # The task doesn't forbid randomness, but deterministic is safer.
    # If I use a fixed seed, I can pre-compute the best h offline?
    # I don't have offline computation. I must do it inside.
    
    rng = np.random.default_rng(42)
    
    for _ in range(500):
        # Perturb
        h_candidate = best_h + rng.normal(0, 0.1, N)
        val = get_val(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate
        # Also try small steps
        h_candidate = best_h + rng.normal(0, 0.01, N)
        val = get_val(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate
            
    return best_h
