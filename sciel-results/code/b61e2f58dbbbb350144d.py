import numpy as np

N = 400

def construct():
    # Sparse combinatorial design:
    # Select a small subset of K intervals to have non-zero values.
    # This decouples the search from the full 400-dimensional space.
    # We initialize with a random permutation to select indices.
    # We use random signs to minimize constructive interference in sidelobes.
    
    K = 10  # Small subset size
    
    # Select K unique indices from a random permutation
    perm = np.random.permutation(N)
    indices = perm[:K]
    
    # Initialize the step function to zeros
    x = np.zeros(N)
    
    # Assign magnitudes to normalize sum of squares to 1
    # x_i^2 = 1/K => x_i = 1/sqrt(K)
    magnitude = 1.0 / np.sqrt(K)
    
    # Assign random signs
    signs = np.random.choice([-1, 1], size=K)
    
    # Set the values
    x[indices] = signs * magnitude
    
    return x
