import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on the Probability of Being the Global Minimum (PBM).
    This approximates a Knowledge Gradient by sampling from the posterior
    and estimating the likelihood that each candidate is the global minimum.
    """
    N = len(mu)
    K = 100  # Number of samples
    
    # Generate samples from the posterior
    # Z shape: (K, N)
    Z = np.random.randn(K, N)
    
    # Y shape: (K, N)
    # Y[k, i] is the sampled value at candidate i for sample k
    Y = mu[None, :] + sigma[None, :] * Z
    
    # Find the index of the minimum value for each sample
    min_indices = np.argmin(Y, axis=1)
    
    # Count how often each candidate was the minimum
    counts = np.bincount(min_indices, minlength=N)
    
    # Normalize to get probabilities
    scores = counts / K
    
    return scores
