import numpy as np

N = 400

# EVOLVE-BLOCK-START

def construct():
    # Choose a small bandwidth W to concentrate the spectrum of h near 0.
    # This makes the spectrum of the autoconvolution C = h*h even more concentrated,
    # resulting in a flatter time-domain autoconvolution, thus minimizing the peak.
    W = 0.05
    
    # Construct the Toeplitz matrix for the Slepian sequence
    indices = np.arange(N)
    diff = np.abs(indices[:, None] - indices[None, :])
    
    # Calculate the sinc kernel
    # np.sinc(x) = sin(pi*x)/(pi*x)
    kernel = np.sinc(diff * W)
    
    # Find the eigenvector corresponding to the largest eigenvalue
    # The matrix is symmetric, so we use eigh
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    
    # The largest eigenvalue is the last one
    h = eigenvectors[:, -1]
    
    # Ensure the sum is positive (arbitrary sign of eigenvector)
    if np.sum(h) < 0:
        h = -h
        
    return h

# EVOLVE-BLOCK-END
