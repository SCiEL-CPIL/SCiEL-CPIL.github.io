import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Structural Change: Thinning-based Expected Improvement (EI) acquisition function
    # This approximates Knowledge Gradient by modeling the probability that a candidate
    # remains the best after selecting the top-k candidates for evaluation.
    
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Calculate the standardized improvement
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Calculate EI
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Thinning-based approximation:
    # The idea is that if we select the top-k candidates, the expected improvement
    # from the best among them is higher than any single candidate's EI.
    # We approximate this by subtracting a "thinning penalty" that accounts for
    # the probability that other candidates might outperform this one.
    
    # For simplicity and efficiency, we use a heuristic:
    # 1. Compute the EI for all candidates
    # 2. Identify the top-k candidates (k=5) based on EI
    # 3. For each candidate, estimate how likely it is to remain the best
    #    among the top-k, and adjust its score accordingly
    
    k = 5
    n = len(ei)
    
    # Get the top-k indices by EI
    top_k_indices = np.argpartition(ei, -k)[-k:]
    top_k_ei = ei[top_k_indices]
    top_k_max_ei = np.max(top_k_ei) if k > 0 else 0
    
    # Thinning penalty: for each candidate, estimate the probability that
    # it remains the best among the top-k candidates.
    # We use a simple model: the probability that candidate i is the best
    # among top-k is proportional to its EI relative to the max EI.
    
    # For candidates in the top-k, we apply a bonus (they're likely to be evaluated)
    # For candidates outside the top-k, we apply a penalty (they're less likely to be evaluated)
    
    # Create a mask for top-k candidates
    in_top_k = np.zeros(n, dtype=bool)
    in_top_k[top_k_indices] = True
    
    # Thinning-based score:
    # - For top-k candidates: boost their EI by a factor that accounts for batch selection
    # - For non-top-k candidates: reduce their EI to reflect lower probability of being selected
    
    # The batch selection probability is modeled as:
    # P(selected) = 1 if in top-k, else 0 (deterministic approximation)
    # But we smooth this with a soft assignment based on EI rank
    
    # Soft assignment: use a sigmoid-like function to smoothly transition
    # between "likely to be selected" and "unlikely to be selected"
    
    # Normalize EI to [0, 1] for soft assignment
    ei_min = np.min(ei)
    ei_range = top_k_max_ei - ei_min if top_k_max_ei > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    # Soft selection probability: candidates with higher EI are more likely to be in top-k
    # Use a sigmoid centered at the k-th highest EI
    k_th_ei = np.sort(ei)[-k] if k <= n else ei_min
    k_th_norm = (k_th_ei - ei_min) / ei_range
    
    # Soft probability of being in top-k
    selection_prob = 1.0 / (1.0 + np.exp(-10 * (ei_norm - k_th_norm)))
    
    # Thinning-adjusted EI:
    # The expected improvement from a batch of k evaluations is higher than
    # any single evaluation. We model this as:
    # EI_batch ≈ EI_single * (1 + alpha * (1 - 1/k))
    # where alpha controls the batch bonus strength
    
    alpha = 0.3  # Batch bonus strength
    
    # Apply batch bonus weighted by selection probability
    thinning_bonus = alpha * (1.0 - 1.0/k) * selection_prob
    ei_thinned = ei * (1.0 + thinning_bonus)
    
    # Cost-aware score: divide by cost^0.8 to soften the cost penalty
    # This allows the acquisition function to favor points with higher expected
    # improvement even at slightly higher cost, helping escape local optima
    kg_score = ei_thinned / (safe_cost ** 0.8)
    
    # Ensure all scores are finite
    kg_score = np.where(np.isfinite(kg_score), kg_score, 0.0)
    
    return kg_score
