"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Hypervolume-based acquisition for minimization of mu and maximization of sigma.
    # Scores candidates by their marginal contribution to the hypervolume of the
    # Pareto front of (mu, sigma) points.
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Fallback LCB if time is tight or inputs are degenerate
    def fallback_lcb():
        kappa = 1.2
        scores = -mu + kappa * sigma
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if time.time() > deadline:
        return fallback_lcb()
        
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Normalize mu and sigma to [0, 1] for fair hypervolume calculation
    # mu: lower is better -> map min to 0, max to 1
    # sigma: higher is better (exploration) -> map min to 0, max to 1
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    
    # Handle constant cases
    if mu_max - mu_min < 1e-12:
        mu_norm = np.zeros(n)
    else:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
        
    if sigma_max - sigma_min < 1e-12:
        sigma_norm = np.zeros(n)
    else:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
        
    # Check deadline
    if time.time() > deadline:
        return fallback_lcb()
        
    # Find non-dominated points
    # Dominance: i dominates j if mu_norm[i] <= mu_norm[j] and sigma_norm[i] >= sigma_norm[j]
    # (and at least one strict)
    # Efficient way: Sort by mu_norm ascending. Keep track of max sigma_norm seen so far.
    # Points that are not dominated are those where sigma_norm is strictly greater than 
    # the max sigma_norm of all points with smaller mu_norm.
    
    # Sort indices by mu_norm
    sorted_indices = np.argsort(mu_norm)
    sorted_mu = mu_norm[sorted_indices]
    sorted_sigma = sigma_norm[sorted_indices]
    
    # Compute max sigma from the right? No, dominance is:
    # A point is dominated if there exists another point with mu <= mu_i and sigma >= sigma_i.
    # Since we sort by mu ascending, for a point at index k in sorted array,
    # it is dominated if there is any point j < k with sigma[j] >= sigma[k].
    # Or if there is a point j > k? No, mu[j] > mu[k], so j cannot dominate k on mu (unless equal).
    # If mu[j] == mu[k], the one with higher sigma dominates.
    
    # Let's find the Pareto front indices in the sorted array.
    # We traverse from left to right (increasing mu).
    # We maintain the maximum sigma seen so far.
    # If current sigma > max_sigma_so_far, it's a candidate for Pareto front (not dominated by previous).
    # But wait, a point with higher mu can dominate a point with lower mu? No.
    # A point with lower mu and higher sigma dominates.
    # So, if we sort by mu ascending, a point i is dominated if there exists j < i with sigma[j] >= sigma[i].
    # This is equivalent to: sigma[i] < max(sigma[0...i-1]).
    
    max_sigma_so_far = -1.0
    pareto_mask_sorted = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if sorted_sigma[i] > max_sigma_so_far + 1e-12:
            pareto_mask_sorted[i] = True
            max_sigma_so_far = sorted_sigma[i]
            
    # Note: Points with equal mu?
    # If mu[i] == mu[j], the one with higher sigma dominates.
    # The loop above handles this: if sigma[i] is not strictly greater than max_so_far, it's marked False.
    # If sigma[i] == max_so_far, it's marked False (dominated by previous equal mu, higher/equal sigma).
    # This is correct.
    
    # Calculate marginal hypervolume contribution for points on the Pareto front
    # The Pareto front in (mu, sigma) space (min mu, max sigma) is a decreasing step function.
    # Points are sorted by mu ascending.
    # Let the Pareto points be P_1, P_2, ..., P_k.
    # P_1 has lowest mu, highest sigma.
    # P_k has highest mu, lowest sigma (among Pareto points).
    
    # Reference point for hypervolume:
    # We use the worst point in the Pareto front as the reference for the "box".
    # Ref_mu = max(mu_norm[Pareto])
    # Ref_sigma = min(sigma_norm[Pareto])
    # Note: In normalized space, Ref is (1, 0) effectively?
    # Actually, the hypervolume is the area "above" the Pareto front and "below" the reference point?
    # Standard definition: Area dominated by the front relative to reference r.
    # Since we want to MAXIMIZE the volume added, we look at the area between the front and the reference.
    # Ref = (max_mu_pareto, min_sigma_pareto).
    # Wait, if we use the worst point in the *current set* as reference, the reference moves.
    # This can cause instability.
    # Alternative: Use fixed reference (1, 0) in normalized space?
    # If we use (1,0), the hypervolume is the area under the front from x=0 to 1, y=0 to 1?
    # No, the front is in [0,1]x[0,1].
    # The area "dominated" by the front (points better than front) is the area above the front?
    # Min mu, Max sigma.
    # "Better" means smaller mu, larger sigma.
    # So the region of interest is top-left.
    # The Pareto front separates the "good" region (top-left) from "bad" (bottom-right).
    # Hypervolume is usually the area of the union of boxes from each point to the reference (bottom-right).
    # So we calculate area from points to (1,0)?
    # Let's stick to the marginal contribution formula derived earlier.
    
    if np.sum(pareto_mask_sorted) == 0:
        return fallback_lcb()
        
    pareto_indices = np.where(pareto_mask_sorted)[0]
    p_mu = sorted_mu[pareto_indices]
    p_sigma = sorted_sigma[pareto_indices]
    
    # Reference point: The worst point in the Pareto set (max mu, min sigma)
    ref_mu = p_mu[-1]
    ref_sigma = p_sigma[-1]
    
    # If ref is (0,0) or similar, handle carefully.
    # Calculate marginal contribution for each Pareto point.
    # Points are sorted by mu ascending.
    # Contribution of P_i:
    # Width = mu_{i+1} - mu_i (for i < last)
    #        ref_mu - mu_i (for i == last)
    # Height = sigma_i - ref_sigma
    
    scores = np.zeros(n)
    
    for i in range(len(pareto_indices)):
        idx_in_sorted = pareto_indices[i]
        idx_original = sorted_indices[idx_in_sorted]
        
        current_mu = p_mu[i]
        current_sigma = p_sigma[i]
        
        if i < len(pareto_indices) - 1:
            next_mu = p_mu[i + 1]
            width = next_mu - current_mu
        else:
            width = ref_mu - current_mu
            
        height = current_sigma - ref_sigma
        
        if width > 1e-12 and height > 1e-12:
            contribution = width * height
            scores[idx_original] = contribution
        else:
            scores[idx_original] = 0.0
            
    # Scale scores to be comparable to LCB?
    # LCB scores are in objective units. Hypervolume scores are in normalized units (0-1).
    # We should scale the hypervolume scores to have a similar magnitude to LCB.
    # Typical LCB score magnitude: mu is ~0-30, sigma ~0-10.
    # Let's scale hypervolume scores by a factor, e.g., 10.0, to make them competitive.
    # Or, we can just return the hypervolume scores directly if we assume the evaluator
    # just picks the argmax. The relative ordering is what matters.
    # However, if we mix with other strategies or if the evaluator does something else,
    # magnitude might matter. The task says "Return 2000 finite scores, higher meaning evaluate next".
    # So relative ordering is key.
    
    # To be safe, let's blend with a small LCB component to break ties and ensure
    # that if hypervolume is 0 for all (degenerate), we still have a gradient.
    # But the directive said "Replace".
    # Let's stick to pure Hypervolume but ensure finite values.
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # If all scores are 0, fallback to LCB
    if np.all(scores == 0):
        return fallback_lcb()
        
    return scores
# EVOLVE-BLOCK-END
