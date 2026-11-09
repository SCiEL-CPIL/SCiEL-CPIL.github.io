import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling with clipping at 0.
    # The objective function is a sphere (x^2), so f(x) >= 0.
    # Predicted values below 0 are physically impossible and likely due to 
    # GP calibration issues (e.g., standardization offset).
    # Clipping the samples at 0 prevents the algorithm from being misled by 
    # negative predictions, focusing exploration on regions where the 
    # function is likely to be low (close to 0) but not unphysically negative.
    
    eps = np.random.standard_normal(mu.shape)
    samples = mu + sigma * eps
    
    # Enforce non-negativity
    samples = np.maximum(samples, 0.0)
    
    # Higher score is better. We want to minimize f, so we prefer lower samples.
    # Thus, score is negative of the sample.
    return -samples
