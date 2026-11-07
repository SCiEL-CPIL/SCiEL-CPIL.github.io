# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) acquisition with sublinear cost discount.
    
    The KG estimates the expected improvement in the global optimum by
    approximating the probability that each candidate is the new minimum.
    We use a simplified one-step KG: for each candidate x_i, the KG value is
    E[max(f_best, f(x_i)) - f_best] = EI(x_i), but we also account for the
    fact that evaluating x_i reduces uncertainty in neighboring points.
    
    To capture this, we add a "knowledge" term that rewards candidates where
    the GP mean is low AND the uncertainty is high (indicating potential to
    reveal a better minimum). This is scaled by 1/sqrt(cost) to balance
    exploration of high-cost regions against budget efficiency.
    
    The full score is: (EI + lambda * knowledge) / sqrt(cost)
    where knowledge = -mu * sigma (low mean, high uncertainty = informative)
    """
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) / np.sqrt(cost) + 0.5 * sigma / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Sublinear cost discount: 1/sqrt(cost) allows exploration of high-cost regions
    cost_discount = 1.0 / np.sqrt(cost)

    # Expected Improvement (minimization: improvement = f_best - f(x))
    # EI = sigma * (z * Phi(z) + phi(z)) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.maximum(ei, 0.0)

    # Knowledge term: rewards candidates with low predicted mean and high uncertainty.
    # This captures the value of reducing uncertainty in potentially good regions.
    # We normalize by f_best scale to make it comparable to EI.
    f_scale = max(abs(f_best), 1e-6)
    knowledge = (-mu / f_scale) * sigma

    # Combine EI and knowledge with a weight on the knowledge term
    w_kg = 0.3
    scores = (ei + w_kg * knowledge) * cost_discount

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
