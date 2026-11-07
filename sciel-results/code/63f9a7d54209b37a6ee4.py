# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 325.0
_HARTMANN_6D_MIN = -3.32237

def _ei_score(mu, sigma, f_best, xi=0.0):
    """Compute Expected Improvement scores."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei

def _pi_score(mu, sigma, f_best):
    """Compute Probability of Improvement scores."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi

def _ucb_score(mu, sigma, f_best, beta):
    """Compute Upper Confidence Bound scores for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    return (f_best - mu) + beta * sigma_safe

def _hybrid_score(mu, sigma, f_best, w_ei, w_ucb, w_pi):
    """Compute a weighted hybrid score."""
    sigma_safe = np.maximum(sigma, 1e-12)
    ei = _ei_score(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, f_best, beta=1.0) # Base UCB
    pi = _pi_score(mu, sigma, f_best)
    # Normalize components roughly to similar scales to make weights meaningful
    # EI and UCB are in objective units. PI is probability [0,1].
    # Scale PI by a typical improvement scale (e.g., 1.0 or range of mu)
    # For robustness, we just use raw weights but PI is much smaller.
    # Let's scale PI by the current gap magnitude to make it comparable.
    gap_scale = max(abs(f_best - np.min(mu)), 1e-6)
    return w_ei * ei + w_ucb * ucb + w_pi * gap_scale * pi

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a meta-search over candidate acquisition strategies.
    Selects the strategy that maximizes the Expected Improvement of the recommended point.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 3.0:
        # Fallback to a robust simple UCB
        sigma_safe = np.maximum(sigma, 1e-12)
        return (f_best - mu) + 1.0 * sigma_safe

    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate candidate strategies
    # Each strategy is a tuple (name, scores_array)
    candidates = []
    
    # 1. Pure EI
    candidates.append(("EI_0", _ei_score(mu, sigma, f_best, xi=0.0)))
    candidates.append(("EI_0.01", _ei_score(mu, sigma, f_best, xi=0.01)))
    candidates.append(("EI_0.05", _ei_score(mu, sigma, f_best, xi=0.05)))
    
    # 2. Pure PI
    candidates.append(("PI", _pi_score(mu, sigma, f_best)))
    
    # 3. UCB with various betas
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for b in betas:
        candidates.append((f"UCB_{b}", _ucb_score(mu, sigma, f_best, b)))
        
    # 4. Hybrids (EI + UCB component)
    # The hybrid score formula: w1*EI + w2*UCB
    # We treat UCB part as exploration term.
    # Note: _ucb_score includes (f_best - mu). EI also includes (f_best - mu).
    # So Hybrid = (w1 + w2)*(f_best - mu) + w1*EI_term + w2*beta*sigma
    # This is equivalent to a shifted UCB/EI mix.
    
    # 5. Knowledge Gradient Approximation
    # KG(x) is the expected reduction in f_best.
    # For a single candidate, KG(x) ~ EI(x).
    # But we can approximate the "value" of a strategy by the KG of its argmax.
    # We will use EI as the quality metric for selection, which is effectively
    # selecting the strategy that points to the highest EI location.
    
    best_strategy_scores = None
    best_quality = -np.inf
    
    # Pre-compute EI for all points to use as the quality metric
    # This is the EI of the point that the strategy would pick.
    # We need to be careful: computing EI for all points is O(N).
    # Finding argmax for each strategy is O(N * K_strategies).
    # Evaluating EI at the argmax is O(1) per strategy if we have pre-computed EI array.
    
    ei_all = _ei_score(mu, sigma, f_best, xi=0.0)
    
    for name, scores in candidates:
        # Ensure scores are finite
        if not np.all(np.isfinite(scores)):
            continue
            
        # Find the best candidate according to this strategy
        idx = np.argmax(scores)
        
        # Evaluate the quality of this choice using the EI of that specific point
        # This aligns the strategy selection with the goal of maximizing expected improvement
        quality = ei_all[idx]
        
        # To break ties or handle cases where EI is 0 (e.g., very early),
        # we can add a small penalty for strategies that pick points with high sigma
        # if EI is low, but usually EI is the right metric.
        # However, if EI is 0 for all (unlikely), we fall back to UCB-like behavior.
        
        if quality > best_quality:
            best_quality = quality
            best_strategy_scores = scores
            best_name = name

    # If no valid strategy found (should not happen), fallback
    if best_strategy_scores is None:
        best_strategy_scores = _ucb_score(mu, sigma, f_best, 1.0)
        
    # Ensure finite output
    best_strategy_scores = np.where(np.isfinite(best_strategy_scores), best_strategy_scores, 0.0)
    
    return best_strategy_scores

# EVOLVE-BLOCK-END
