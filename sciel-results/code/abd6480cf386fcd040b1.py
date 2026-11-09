"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition with deterministic fallback.
    
    Samples the GP posterior at each candidate: y_i = mu_i + sigma_i * z_i
    where z_i ~ N(0, 1). We want to MINIMIZE, so higher acquisition score
    means more promising (lower expected value with uncertainty exploration).
    
    We use a stochastic ranking: for each candidate, compute a sample from the
    posterior, then score = -sampled_value (so lower sampled value = higher score).
    To make this deterministic and reproducible per call, we use a fixed random
    draw per call. However, the evaluator calls this multiple times with fresh
    candidates, so we want the scores to be deterministic given mu, sigma, f_best.
    
    Actually, the task says "Return 2000 finite scores, higher meaning evaluate next."
    The evaluator picks the candidate with the highest score. We need deterministic
    scores (no randomness that would change between calls with same inputs, or it
    might be okay if it's just one call per step). Let me re-read: "at each step draws
    2000 fresh uniform candidates, computes the posterior there, calls acquisition and
    evaluates the candidate with the highest score."
    
    So it's one call per step. We can use randomness. But to be safe and to make
    it a proper acquisition function that's deterministic in structure, let's use
    a hybrid: Thompson sampling idea but implemented as a deterministic score that
    approximates the expected minimum of the posterior.
    
    Actually, let me think about this differently. The directive says to use Thompson
    Sampling-inspired acquisition. The key insight is: instead of computing EI (which
    is a deterministic function of mu, sigma, f_best), we can create a score that
    naturally balances exploration and exploitation by treating the posterior sample
    as the decision criterion.
    
    For a deterministic score that captures Thompson sampling's behavior:
    - The expected minimum of mu + sigma*z over z~N(0,1) is mu - sigma * E[max(0,z)]... 
      no, that's not quite right.
    
    Let me think: if we sample y = mu + sigma*z, and we want to pick the point with
    lowest y, the "score" for a Thompson sampling approach would be based on the
    distribution of y. A common deterministic approximation is to use the mean of the
    lower tail, or to use a quantile.
    
    A simpler and effective approach: use the expected value of the negative of the
    posterior sample, which is just -mu. But that ignores uncertainty.
    
    Better: use a score that combines exploitation (low mu) and exploration (high sigma).
    Thompson sampling naturally does this because a high-sigma point has a chance of
    sampling very low. The expected minimum over samples isn't straightforward, but
    we can approximate: the probability that a sample is below f_best is P(mu + sigma*z < f_best)
    = P(z < (f_best - mu)/sigma) = Phi((f_best - mu)/sigma).
    
    A good deterministic proxy for Thompson sampling is to use the expected value of
    min(y, f_best) or similar. But let's use a well-known approach:
    
    Score = -mu + beta * sigma * norm.pdf(z) / norm.cdf(z) for some beta...
    
    Actually, let me just use a clean Thompson-sampling-inspired deterministic score:
    
    For each candidate, the posterior is N(mu_i, sigma_i^2). In Thompson sampling,
    we'd sample y_i ~ N(mu_i, sigma_i^2) and pick argmin y_i. The deterministic
    equivalent that preserves the ranking properties is to use the mean of the
    distribution shifted by an exploration term.
    
    A practical and effective choice: use the expected value of the negative sample,
    weighted by how likely it is to beat f_best. This gives:
    
    score_i = -(mu_i) + sigma_i * norm.pdf(z_i) / max(norm.cdf(z_i), 1e-9)
    
    where z_i = (f_best - mu_i) / sigma_i. This is related to the "expected
    improvement" structure but with a different balance.
    
    Actually, let me just use a simpler and proven approach: the score for Thompson
    sampling in a deterministic setting can be approximated by the lower quantile of
    the posterior. For example, the 10th percentile of N(mu, sigma^2) is
    mu + sigma * norm.ppf(0.1) = mu - 1.28 * sigma.
    
    So score = -(mu - 1.28 * sigma) = -mu + 1.28 * sigma.
    
    This naturally balances: low mu is good (exploitation), high sigma is good
    (exploration). The 1.28 factor corresponds to the 10th percentile, which is
    a reasonable exploration level.
    
    Let me use this simple, clean approach.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score relative to best
    z = (f_best - mu) / sigma_safe
    
    # Thompson sampling inspired: use a quantile-based score
    # The 10th percentile of the posterior is a good exploration/exploitation balance
    # score = -(mu + sigma * norm.ppf(0.1)) = -mu + 1.28 * sigma
    # But let's make it adaptive: when f_best is close to mu, explore less;
    # when f_best is far, explore more.
    
    # Use a blend: primary term is exploitation (-mu), secondary is exploration
    # The exploration term is scaled by how uncertain we are relative to the gap
    gap = np.maximum(f_best - mu, 0)
    exploration_weight = sigma_safe / (gap + sigma_safe + 1e-9)
    
    # Score: higher is better
    # Base: -mu (exploit low predicted values)
    # Exploration bonus: proportional to sigma, scaled by uncertainty relative to gap
    score = -mu + 1.5 * sigma_safe * exploration_weight
    
    # Add a small EI-like term for fine-tuning
    ei_term = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    score = 0.5 * score + 0.5 * ei_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
