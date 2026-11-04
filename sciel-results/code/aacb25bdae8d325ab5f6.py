import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse signed step function to minimize the autoconvolution peak.
    Uses a random subset of intervals with random signs, scaled to reduce the
    autoconvolution peak through destructive interference.
    """
    # Select a sparse subset of intervals using random permutation
    M = 50  # Number of active intervals
    perm = np.random.permutation(N)
    active_indices = perm[:M]
    
    # Initialize with zeros
    x = np.zeros(N)
    
    # Assign random signs to active intervals, scaled by 1/sqrt(N)
    signs = np.random.choice([-1.0, 1.0], size=M)
    x[active_indices] = signs / np.sqrt(N)
    
    return x
# EVOLVE-BLOCK-END
