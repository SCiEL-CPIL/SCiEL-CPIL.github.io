"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = time.time() + 325  # 5s safety margin before 330s kill


def _ensure_deadline():
    """Raise if we're close to the deadline."""
    if time.time() > _deadline:
        raise TimeoutError("Deadline exceeded")


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-based acquisition function for Goldstein-Price 2D.
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior std
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    _ensure_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: sample from posterior predictive and pick the best
    # Since we can only return a score array (not pick a point), we use an
    # analytical approximation: the expected maximum improvement over a
    # Thompson sample, which relates to EI but with a different weighting.
    
    # For Thompson Sampling, the score at each point is the probability that
    # a sample from N(mu_i, sigma_i) beats f_best, weighted by the expected
    # improvement conditional on beating f_best.
    
    # P(sample < f_best) = Phi((f_best - mu) / sigma)  [minimization]
    # E[min(sample, f_best)] = mu * Phi(z) + sigma * phi(z) + f_best * (1 - Phi(z))
    #   where z = (f_best - mu) / sigma
    # E[max(0, f_best - sample)] = EI (standard)
    
    # A good TS-like score: use the expected improvement but with a
    # different normalization that emphasizes exploration when uncertainty is high.
    # Specifically, use: score = E[max(0, f_best - X)] / sigma + c * sigma
    # This is NEI + c*sigma, which balances exploitation and exploration.
    
    # Actually, for true Thompson Sampling behavior, we want the score to
    # reflect the "value" of sampling at that point. The optimal TS policy
    # picks argmin of a sample from the posterior. Since we return scores,
    # we use the expected value of the minimum of K samples from the posterior
    # at each point (for some K), which naturally balances exploration/exploitation.
    
    # For K=1, this is just mu. For larger K, it favors points with high sigma.
    # E[min of K samples from N(mu, sigma)] = mu - sigma * E[min of K standard normals]
    # E[min of K standard normals] is a constant depending on K.
    
    # Let's use a blend: primarily EI-based (which is what worked well before)
    # but with a Thompson-Sampling-inspired exploration bonus.
    
    # Standard EI
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Thompson Sampling score: approximate as the expected reduction in
    # f_best if we sample here. This is related to EI but let's use a
    # different formulation that emphasizes exploration more.
    
    # Score = P(beat f_best) * E[improvement | beat f_best] + exploration_bonus
    # = EI + exploration_bonus
    
    # The exploration bonus should be proportional to sigma but scaled
    # by how "promising" the region is (not just pure UCB).
    # Use: bonus = sigma * w(z) where w(z) is a function that is large
    # when z is moderate (uncertain but not hopeless) and small when
    # z is very negative (clearly worse than f_best) or very positive
    # (clearly better, so exploitation dominates).
    
    # w(z) = pdf(z) / (cdf(z) + 1-cdf(z)) ... no, that's just pdf(z)
    # Let's use: w(z) = exp(-z^2/2) * (1 - cdf(z)) for z > 0 (exploration in uncertain territory)
    # and w(z) = exp(-z^2/2) * cdf(z) for z < 0 (exploitation)
    # Actually, simpler: just use pdf(z) as the exploration weight.
    
    # pdf(z) is maximized at z=0 (maximum uncertainty about whether we beat f_best)
    # and decays for |z| >> 1. This is a natural TS-like exploration bonus.
    
    pdf_z = norm.pdf(z)
    
    # Scale the exploration bonus to be comparable to EI
    # EI has units of the objective, sigma has units of the objective
    # pdf_z is dimensionless
    # So sigma * pdf_z has the right units
    
    # Weight: start with more exploration early, less later
    # Use the fraction of points with positive EI as a proxy for progress
    n_positive = np.sum(ei > 1e-10)
    frac_positive = n_positive / len(ei) if len(ei) > 0 else 0.5
    
    # Beta transitions from ~1.5 (early, explore more) to ~0.3 (late, exploit more)
    beta = 0.3 + 1.2 * (1.0 - frac_positive)
    
    # Final score: EI + beta * sigma * pdf(z)
    # The sigma*pdf(z) term is the "Thompson Sampling exploration bonus"
    scores = ei + beta * sigma_safe * pdf_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    _ensure_deadline()
    
    return scores
# EVOLVE-BLOCK-END
