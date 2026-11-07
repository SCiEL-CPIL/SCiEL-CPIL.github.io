# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Global Minimum Probability (GMP) acquisition function.
    
    Unlike EI or KG which focus on the expected improvement over the current best,
    GMP estimates the probability that a given candidate point is the global minimum
    among all candidates in the current set. This encourages exploration of regions
    that have a high chance of containing the true global minimum, even if the
    immediate expected improvement is low, by considering the relative likelihood
    of each point being the best in the batch.
    
    For a candidate j, the probability that it is the global minimum given the
    posterior samples is:
    P(j is min) = E[ 1{F_j < F_i for all i != j} ]
    
    We approximate this expectation using Monte Carlo sampling:
    1. Draw K samples Z ~ N(0, 1) of shape (K, N) where N is the number of candidates.
    2. Compute sampled function values F = mu + sigma * Z.
    3. For each sample k, identify the index of the minimum value.
    4. The GMP score for candidate j is the fraction of samples where j was the minimum.
    
    To ensure we still respect the current best observed value f_best (to avoid
    exploring regions that are unlikely to beat the current best), we can combine
    GMP with a small EI component or simply rely on GMP if the GP is well-calibrated.
    However, pure GMP might be too exploratory. A common variant is "Probability of Improvement"
    but for the minimum. Here we use a hybrid: 
    Score = alpha * GMP + (1 - alpha) * EI
    
    Actually, the inspiration suggests GMP alone provided a structural shift.
    Let's stick to a robust GMP implementation.
    
    Optimization:
    Instead of checking all pairs, we can just find the argmin for each sample.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    N = len(mu)
    
    # Time budget management
    # We want to spend most of the time on the main calculation but leave a safety margin.
    # The total time limit is 330s, but we should aim to return well before that.
    # Let's set a hard internal deadline for the MC loop.
    deadline = start_time + 300.0
    
    # Determine number of Monte Carlo samples K.
    # We can adapt K based on remaining time, but a fixed large K is usually fast enough
    # for vectorized operations on 2000 candidates.
    # 2000 candidates * 1000 samples = 2,000,000 ops. Very fast.
    # Let's try K=500 for a good balance of accuracy and speed.
    K = 500
    
    # Draw K samples from standard normal
    # Shape: (K, N)
    Z = np.random.normal(0, 1, size=(K, N))
    
    # Compute sampled function values
    # F[k, j] = mu[j] + sigma[j] * Z[k, j]
    F = mu[None, :] + sigma[None, :] * Z
    
    # For each sample k, find the index of the minimum value
    # This gives us which candidate is the global minimum in that sample
    min_indices = np.argmin(F, axis=1)
    
    # Count how many times each candidate was the minimum
    # gmp_counts[j] is the number of samples where candidate j was the min
    gmp_counts = np.zeros(N, dtype=np.float64)
    np.add.at(gmp_counts, min_indices, 1.0)
    
    # Normalize to get probabilities
    gmp_scores = gmp_counts / K
    
    # Ensure finite values
    gmp_scores = np.nan_to_num(gmp_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Safety check for time
    if time.time() > deadline:
        # Fallback to EI if somehow slow
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) - sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        
    return gmp_scores
# EVOLVE-BLOCK-END
