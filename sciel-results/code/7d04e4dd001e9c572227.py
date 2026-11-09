import numpy as np

def acquisition(mu, sigma, f_best):
        # Multi-sample Thompson Sampling
        # Draw K samples to reduce variance in the selection
        K = 5
        n = mu.shape[0]
        # Generate K x N random normals
        eps = np.random.standard_normal((K, n))
        # Compute samples: (K, N)
        samples = mu[None, :] + sigma[None, :] * eps
        
        # Use the mean of the samples as the representative value
        # This provides a more stable estimate than a single sample
        mean_samples = np.mean(samples, axis=0)
        
        # Higher score means lower objective value
        return -mean_samples
