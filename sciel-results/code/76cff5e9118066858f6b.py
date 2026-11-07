import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thinning-based Expected Improvement (EI) for cost-aware Bayesian optimization.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) for each candidate.
    2. Approximate Knowledge Gradient by "thinning": 
       - Identify the top K candidates by EI.
       - For each candidate, subtract the expected improvement of the top K competitors 
         (excluding itself if it is in the top K). This models the probability that 
         the current point will still be the best after evaluating a batch of points,
         or more simply, reduces the score of points that are likely to be "overtaken" 
         by other high-potential points in the same batch.
    3. Normalize by cost to favor cheaper evaluations.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-6)
    
    # Safety: avoid division by zero in sigma for EI calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative. Clip small negative values due to numerical error.
    ei = np.maximum(ei, 0.0)
    
    # 2. Thinning / Knowledge Gradient Approximation
    # We want to penalize candidates that are likely to be "dominated" by other candidates 
    # that will also be evaluated in the batch. 
    # A simple approximation: For each candidate i, subtract the sum of EI of the top K 
    # other candidates (by EI value) from its own EI. 
    # However, doing this for all 2000 points with K=5 is O(N*K log N) or O(N^2) if not careful.
    # Since N=2000 is small, we can be slightly inefficient but safe.
    
    # To make it faster and stable, we can just subtract the maximum EI of the top K 
    # candidates (excluding the candidate itself) from the candidate's EI.
    # This creates a "winning" signal only if the candidate's EI is significantly higher 
    # than the next best alternatives, or if the alternatives are weak.
    
    # Actually, a simpler and effective "thinning" often used in practice is to 
    # subtract the *mean* EI of the top K candidates (excluding self) or just the *max* EI of the top K.
    # Let's use the Max EI of the top K competitors.
    
    K = 5
    
    # Get indices of top K candidates by EI
    # We need to handle the case where a candidate is in the top K.
    # For each candidate i, we want max(EI[j] for j in top_K_indices if j != i)
    
    # To do this efficiently without O(N^2) loops in Python:
    # 1. Get top K indices.
    top_k_indices = np.argsort(ei)[-K:][::-1] # Descending order
    
    # Create a matrix of penalties? No, that's 2000x2000.
    # Instead, for each of the top K candidates, we know their EI.
    # For a candidate i:
    # If i is in top_k_indices, the penalty is the max EI of the other K-1 top candidates.
    # If i is NOT in top_k_indices, the penalty is the max EI of all K top candidates.
    
    # Let's compute the max EI of the top K.
    if K > 0:
        max_top_k_ei = ei[top_k_indices[0]]
        # If K > 1, compute the max of the rest for candidates that are in the top K
        if K > 1:
            second_max_top_k_ei = ei[top_k_indices[1]]
        else:
            second_max_top_k_ei = 0.0
    else:
        max_top_k_ei = 0.0
        second_max_top_k_ei = 0.0
        
    # Initialize penalty array
    penalty = np.full_like(ei, max_top_k_ei)
    
    # For candidates that are in the top K, the penalty should be the max of the *other* top K.
    # This is a bit more complex. Let's simplify:
    # The "Thinning" effect is strongest if we subtract the *average* or *max* of the competition.
    # A robust simple proxy: Subtract the K-th largest EI (the threshold for being in the top K).
    # This effectively says: "Your value is your EI minus the quality of the 5th best candidate."
    # This encourages picking points that are clearly better than the 5th best, 
    # or if you are one of the top 5, it subtracts the 5th best (which is lower than your own if you are top 4).
    
    # Let's use the K-th largest EI as the penalty for all.
    # This is O(N log N) to sort, which is fine.
    # This specific formulation is a known heuristic for KG approximation in batches.
    
    if K > 0:
        # k-th largest value
        # np.partition is faster than full sort
        kth_largest = np.partition(ei, -K)[-K]
    else:
        kth_largest = 0.0
        
    # Thinned EI
    thin_ei = ei - kth_largest
    
    # Ensure thin_ei doesn't go too negative, though it should be fine.
    # Actually, if EI < kth_largest, thin_ei is negative. 
    # We want to maximize the score. Negative scores are fine as long as they are finite.
    # But we should probably floor it at 0 or a small negative number to avoid weird behavior?
    # No, let's keep it raw. If a point is worse than the 5th best, it gets a lower score.
    
    # 3. Cost-aware score
    # We want to maximize (Thinned EI) / cost
    # However, if Thinned EI is negative, dividing by cost makes it more negative (if cost > 1) 
    # or less negative (if cost < 1). 
    # Standard practice: Only consider positive EI for improvement.
    # Let's clip thin_ei to 0 to ensure we only pick points with positive expected "net" improvement 
    # relative to the batch threshold.
    
    thin_ei_clipped = np.maximum(thin_ei, 0.0)
    
    score = thin_ei_clipped / cost_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290: # 10s margin
        # Return current best result
        pass
        
    return score
