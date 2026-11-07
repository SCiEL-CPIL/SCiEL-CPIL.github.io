import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian optimization using a batch-aware
    Look-Ahead strategy that simulates the next 5 evaluations by greedily
    selecting the top-scoring candidate and updating the posterior variance
    locally to approximate the information gain.
    
    This approach scores candidates based on their expected contribution to
    reducing the minimum value over the entire batch, not just the immediate
    next evaluation.
    """
    N = len(mu)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of look-ahead steps
    n_lookahead = 5
    
    # Initialize scores as EI for each candidate
    def compute_ei(mu_vals, sigma_vals, f_best_val):
        """Compute Expected Improvement scores."""
        z = (f_best_val - mu_vals) / sigma_vals
        ei = (f_best_val - mu_vals) * norm.cdf(z) + sigma_vals * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Initial EI scores
    ei_scores = compute_ei(mu, sigma, f_best)
    
    # Look-ahead simulation
    # We simulate n_lookahead greedy selections and track how much each candidate
    # contributes to the overall improvement
    
    # Initialize contribution array
    contributions = np.zeros(N)
    
    # Track remaining candidates and their updated sigmas
    remaining_mask = np.ones(N, dtype=bool)
    current_sigma = sigma.copy()
    current_f_best = f_best
    
    for step in range(n_lookahead):
        # Check time budget
        if time.time() > deadline - 5.0:
            break
        
        # Compute EI scores for remaining candidates
        ei_remaining = compute_ei(mu[remaining_mask], current_sigma[remaining_mask], current_f_best)
        
        # Find the best candidate
        if not np.any(remaining_mask):
            break
        
        best_idx_in_remaining = np.argmax(ei_remaining)
        best_idx = np.where(remaining_mask)[0][best_idx_in_remaining]
        
        # Credit this candidate with its EI value
        contributions[best_idx] += ei_remaining[best_idx_in_remaining]
        
        # Simulate evaluating this point
        # Update f_best if this point improves it
        simulated_value = mu[best_idx]  # Approximate with mean
        if simulated_value < current_f_best:
            current_f_best = simulated_value
        
        # Update sigma for nearby points (local variance reduction)
        # Points closer to the selected point get reduced uncertainty
        # Use a simple distance-based decay in the index space
        # Since we don't have coordinates, we approximate by reducing sigma
        # for all remaining points, with stronger reduction for higher EI points
        # (heuristic: points with similar characteristics to the selected point)
        
        # Simple update: reduce sigma for all remaining points by a factor
        # that depends on how much information this evaluation provides
        # The information gain is approximately related to the EI value
        info_gain = ei_remaining[best_idx_in_remaining]
        
        # Reduce sigma for remaining points
        # Use a decay factor based on information gain
        decay_factor = np.exp(-info_gain * 0.1)
        decay_factor = max(decay_factor, 0.5)  # Don't reduce too much
        
        # Apply to remaining candidates only
        indices_remaining = np.where(remaining_mask)[0]
        current_sigma[indices_remaining] *= decay_factor
        current_sigma[indices_remaining] = np.maximum(current_sigma[indices_remaining], 1e-9)
        
        # Remove the selected candidate
        remaining_mask[best_idx] = False
    
    # Final scores: combine initial EI with look-ahead contributions
    # Candidates that are selected in the look-ahead get boosted
    # Candidates that benefit from variance reduction also get boosted
    
    # Compute final EI with updated sigma for all candidates
    final_ei = compute_ei(mu, current_sigma, f_best)
    
    # Combine: base EI + contribution from look-ahead
    # Scale contributions to be comparable to EI
    max_contrib = np.max(contributions) if np.max(contributions) > 0 else 1.0
    scaled_contributions = contributions / max_contrib * max(final_ei) * 0.5
    
    scores = final_ei + scaled_contributions
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
