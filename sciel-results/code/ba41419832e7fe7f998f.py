import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo Knowledge Gradient (MCKG) approximation.
    
    Args:
        mu: numpy array of shape (N,) posterior mean.
        sigma: numpy array of shape (N,) posterior standard deviation.
        f_best: float, the best observed function value so far.
        
    Returns:
        numpy array of shape (N,) scores. Higher scores indicate better candidates.
    """
    N = len(mu)
    if N == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Select Top-K competitors based on lowest mean (most promising)
    # We use K=20 to balance accuracy and speed
    K = 20
    if N > K:
        top_k_indices = np.argsort(mu)[:K]
    else:
        top_k_indices = np.arange(N)
    
    # Pre-sample the competitors for Monte Carlo
    # M samples
    M = 100
    # Shape: (M, K)
    # We need to handle the case where i is in top_k_indices separately
    
    # To optimize, we can sample all candidates at once? No, we need specific combinations.
    # Let's just loop over candidates or vectorize if possible.
    
    # Vectorized approach for MCKG is tricky due to the 'exclude i' logic.
    # Let's use a loop over the 2000 candidates. 2000 * (100 * 20) ops is fine.
    
    scores = np.zeros(N)
    
    # Pre-generate random normal numbers for efficiency?
    # We can generate a large block of random numbers.
    
    for i in range(N):
        # Determine competitors
        if i in top_k_indices:
            # Competitors are top_k_indices excluding i
            comp_indices = [j for j in top_k_indices if j != i]
        else:
            comp_indices = top_k_indices
            
        if len(comp_indices) == 0:
            # Only candidate? Then VOI = f_best - E[f_i] = f_best - mu_i
            scores[i] = f_best - mu[i]
            continue
            
        # Sample M values for candidate i
        # f_i ~ N(mu_i, sigma_i^2)
        f_i_samples = np.random.normal(mu[i], sigma_safe[i], M)
        
        # Sample M values for each competitor
        # We can do this in a loop over competitors or vectorize
        # f_comp ~ N(mu_comp, sigma_comp^2)
        # Shape: (M, num_comps)
        mu_comps = mu[comp_indices]
        sigma_comps = sigma_safe[comp_indices]
        
        # Generate (M, num_comps) samples
        f_comp_samples = np.random.normal(mu_comps, sigma_comps, size=(M, len(comp_indices)))
        
        # Find min among competitors for each sample
        min_comp_samples = np.min(f_comp_samples, axis=1)
        
        # Find min between candidate i and competitors
        min_all_samples = np.minimum(f_i_samples, min_comp_samples)
        
        # Expected minimum
        expected_min = np.mean(min_all_samples)
        
        # Value of Information
        scores[i] = f_best - expected_min
        
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
