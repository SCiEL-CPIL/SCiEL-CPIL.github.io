import numpy as np

N = 400

def construct():
    # New approach: Random signed sequence (Rademacher)
    # This explores the space of low-autocorrelation candidates,
    # although for this specific metric (max autoconv / sum^2),
    # constant sequences are theoretically optimal.
    # We use a deterministic seed for reproducibility.
    rng = np.random.default_rng(42)
    s = rng.choice([-1, 1], size=N)
    
    # Ensure the sum is not zero to avoid division by zero
    # If sum is 0, flip the first element
    if np.sum(s) == 0:
        s[0] = -s[0]
        
    return s
