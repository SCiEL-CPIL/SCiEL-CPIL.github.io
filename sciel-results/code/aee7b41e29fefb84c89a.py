"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function approximation.
    
    The KG for a candidate x_i is the expected reduction in the uncertainty
    of the global minimum if x_i were evaluated.
    KG(x_i) = E[min(f_best, f(x_i))] - E[min(f_best, f(x_1), ..., f(x_N))]
    
    We approximate the expectations using Monte Carlo sampling.
    """
    start_time = time.time()
    # Safety margin: 300s limit. We aim to finish in < 1s.
    
    n_candidates = len(mu)
    
    # Handle edge cases where sigma is 0 or very small
    # If sigma is 0, the value is deterministic.
    # We add a small epsilon to sigma for sampling stability, 
    # but for KG, if sigma is 0, the min is just the value itself.
    
    # Number of Monte Carlo samples. 
    # 100 samples provides a good balance between accuracy and speed.
    n_samples = 100
    
    # Generate samples: shape (n_samples, n_candidates)
    # f_j ~ N(mu_j, sigma_j^2)
    # We use standard normal and scale
    z = np.random.standard_normal((n_samples, n_candidates))
    samples = mu + z * sigma
    
    # Term 1: E[min(f_best, f(x_i))] for each i
    # This is the mean over samples of min(f_best, sample_i)
    # min(f_best, sample) is element-wise
    min_with_best = np.minimum(f_best, samples)
    expected_min_i = np.mean(min_with_best, axis=0)
    
    # Term 2: E[min(f_best, f(x_1), ..., f(x_N))]
    # This is the mean over samples of the minimum of all candidates in that sample, 
    # also capped by f_best.
    # For each sample row, find the min of all candidates.
    min_all_candidates = np.min(samples, axis=1)
    # Cap by f_best
    min_global = np.minimum(f_best, min_all_candidates)
    expected_global_min = np.mean(min_global)
    
    # KG score = expected_min_i - expected_global_min
    # Note: Since we are minimizing, a lower expected global min is better.
    # The KG value represents the *reduction* in the expected minimum.
    # Higher KG means better.
    kg_scores = expected_min_i - expected_global_min
    
    # Ensure finite scores
    kg_scores = np.nan_to_num(kg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return kg_scores
# EVOLVE-BLOCK-END
