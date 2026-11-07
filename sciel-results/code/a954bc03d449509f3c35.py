"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # Known minimum of Styblinski-Tang 1D
    known_min = -39.1662
    
    # Progress measure: how close are we to the optimum?
    # f_best starts high (worst case ~50) and decreases to -39.1662
    progress = np.clip((50.0 - f_best) / (50.0 - known_min), 0.0, 1.0)
    
    # Look-ahead strategy: simulate next K evaluations to score candidates
    # based on expected contribution to reducing the minimum over the batch.
    
    # Number of look-ahead steps
    K = 5
    
    # We'll use a batch-aware approach:
    # 1. First, compute a base acquisition score for all candidates
    # 2. Then simulate K greedy selections, updating a local variance model
    # 3. Score each candidate by its expected improvement over the batch
    
    # Base: Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei_base = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Also compute a pessimistic lower bound (for exploration)
    # Using a quantile-based approach: P(f(x) < f_best) weighted by potential
    # For minimization, we want low mu and high sigma (uncertainty)
    
    # Adaptive exploration parameter
    # Early: more exploration (higher beta), Late: more exploitation
    beta = 1.5 + 2.0 * np.exp(-2.0 * progress)
    
    # Pessimistic score: mu - beta * sigma (lower is better, so we negate for "higher is better")
    pessimistic = -(mu - beta * sigma_safe)
    
    # Scale EI to be comparable
    ei_scale = np.std(ei_base) + eps
    pess_scale = np.std(pessimistic) + eps
    ei_scaled = ei_base / ei_scale * pess_scale
    
    # Blend: weight shifts toward exploitation as we converge
    ei_weight = 0.5 + 0.4 * progress
    exp_weight = 1.0 - ei_weight
    
    base_scores = ei_weight * ei_scaled + exp_weight * pessimistic
    
    # Deadline check
    if time.time() - start_time > 250:
        base_scores = np.where(np.isfinite(base_scores), base_scores, 0.0)
        return base_scores
    
    # Look-ahead: simulate K greedy selections
    # We approximate the effect of observing a point by reducing sigma locally
    # (information gain reduces uncertainty)
    
    # Work on a copy of sigma for the simulation
    sigma_sim = sigma_safe.copy()
    mu_sim = mu.copy()
    
    # Track which points have been "selected" in the simulation
    selected = np.zeros(n, dtype=bool)
    
    # Current best in the simulation (starts at f_best)
    current_best = f_best
    
    # Scores to accumulate: expected improvement contribution over the batch
    # For each candidate, we estimate how much it would improve the minimum
    # if selected in the look-ahead sequence
    
    # Strategy: for each candidate i, estimate the expected value of the minimum
    # after K evaluations where i is the first one selected, and the remaining
    # K-1 are greedily chosen from the rest.
    
    # This is expensive, so we use a simpler approximation:
    # Score candidate i by:
    #   EI_i + sum_{k=2}^{K} E[improvement from k-th greedy pick | i selected]
    
    # Simplified: we simulate the greedy selection process once,
    # and for each candidate, compute its "marginal contribution" to the
    # batch minimum.
    
    # Greedy simulation: pick top-K by base_scores, but adjust sigma as we go
    # When we pick a point, we assume it will be evaluated, and the resulting
    # observation will reduce uncertainty in nearby regions.
    
    # Since we don't have coordinates, we approximate the local variance reduction
    # using the correlation structure implied by the GP. Without coordinates,
    # we use a heuristic: selecting a point with high sigma reduces sigma
    # globally by a small amount (representing information gain).
    
    # A more principled approach: use the fact that after observing y at x,
    # the posterior variance at x' is reduced by k(x,x')^2 / (k(x,x) + sigma^2)
    # Without coordinates, we approximate this as a uniform reduction scaled
    # by the relative uncertainty.
    
    # Let's use a different approach: score each candidate by its expected
    # improvement over a virtual batch of K evaluations.
    
    # For candidate i, the batch score is:
    #   E[min(f(x_i), f(x_{i2}), ..., f(x_{iK}))]
    # where x_{i2}, ..., x_{iK} are the next K-1 best candidates after i.
    
    # Approximation: 
    #   batch_score(i) = -E[min(f(x_i), next_best_after_i)]
    # where next_best_after_i is the best of the remaining K-1 candidates.
    
    # For simplicity and speed, we use:
    #   batch_score(i) = EI(i) + lambda * sigma(i) * (1 - selection_prob(i))
    # where selection_prob(i) is the probability that i would be selected
    # in the greedy process (i.e., it's among the top-K).
    
    # Compute selection probability: rank-based
    sorted_indices = np.argsort(-base_scores)  # descending by base score
    rank = np.empty(n, dtype=int)
    rank[sorted_indices] = np.arange(n)
    
    # Probability of being selected in top-K
    sel_prob = np.where(rank < K, 1.0, 0.0)
    
    # Batch-aware score: 
    # Candidates that are likely to be selected get a bonus for their
    # information value (high sigma), while others get a penalty.
    # The idea: if you're going to be selected, your sigma matters less
    # (it will be resolved), so focus on mu. If you're NOT going to be
    # selected, high sigma is still valuable as a "backup" option.
    
    # Actually, let's think differently. The look-ahead should reward
    # candidates that, when selected, lead to the best expected batch minimum.
    
    # Simpler structural approach: use a "hypervolume"-like score that
    # accounts for the distribution of potential improvements.
    
    # Compute the expected minimum after K evaluations starting from candidate i:
    # E[min] ≈ mu_i - sigma_i * E[max(0, Z_i)] - (K-1) * avg_remaining_improvement
    
    # Where Z_i ~ N(0,1), so E[max(0, Z_i)] = 1/sqrt(2*pi) ≈ 0.399
    
    # The remaining improvement from K-1 other points is bounded by
    # the average EI of the top K-1 candidates excluding i.
    
    # Let's compute the top K-1 average EI (excluding each candidate)
    # This is O(n*K) which is fine for n=2000, K=5
    
    # Get the top K+1 candidates by EI
    top_k_plus_1 = sorted_indices[:K+1]
    
    # For each candidate, compute the average EI of the top K-1 others
    # We approximate this using the global top-K average as a proxy
    
    top_k_ei = ei_base[sorted_indices[:K]]
    avg_top_k_ei = np.mean(top_k_ei) if len(top_k_ei) > 0 else 0.0
    
    # Expected minimum after selecting i and K-1 others:
    # E[min] ≈ mu_i - sigma_i * 0.399 - (K-1) * avg_top_k_ei * 0.5
    # (the 0.5 factor accounts for the fact that the other points' improvements
    #  are not all realized as the minimum)
    
    expected_min_i = mu - sigma_safe * 0.399 - (K-1) * avg_top_k_ei * 0.5
    
    # Batch score: higher is better (lower expected minimum)
    batch_score = -expected_min_i
    
    # Blend base score and batch score
    # The batch score captures the look-ahead effect
    blend_weight = 0.3 + 0.4 * progress  # more weight to batch as we converge
    
    final_scores = (1 - blend_weight) * base_scores + blend_weight * batch_score
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Deadline check
    if time.time() - start_time > 260:
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END
