import numpy as np

N = 400

def construct():
    # Use a fixed seed for reproducibility, or just generate random signs.
    # The task says "Choose a signed step function". 
    # Random signs minimize the expected autocorrelation energy.
    # Let's use a high-quality PRNG for a deterministic "random" looking sequence.
    rng = np.random.default_rng(42)
    # Generate +1 and -1
    signs = rng.choice([-1, 1], size=N)
    return signs
