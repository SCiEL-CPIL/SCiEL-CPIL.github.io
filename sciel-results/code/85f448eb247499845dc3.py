"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 280.0  # Safety margin below 300s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Knowledge Gradient approximation
    with decoupled discovery probability and expected value reduction.
    """
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute z-score: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)

    cdf = norm.cdf(z)   # P(x is a new best)
    pdf = norm.pdf(z)

    # Standard EI: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # Knowledge Gradient core components:
    # 1. Discovery probability: P(f(x) < f_best) = cdf
    # 2. Conditional expected improvement given it IS a new best:
    #    E[f_best - f(x) | f(x) < f_best] = ei / cdf (when cdf > 0)
    #    = (f_best - mu) + sigma * pdf/cdf
    #    The pdf/cdf term is the inverse Mills ratio, which captures
    #    the "tail" expectation beyond the threshold.
    
    # Inverse Mills ratio: E[Z | Z > z] where Z ~ N(0,1), but here we need
    # E[Z | Z < z] = -phi(z)/Phi(z)
    # Actually, for the conditional expectation:
    # E[f_best - f(x) | f(x) < f_best] = (f_best - mu) + sigma * phi(z)/Phi(z)
    # This is the expected improvement GIVEN that x improves on f_best.
    
    # Use a safe version to avoid division by zero
    cdf_safe = np.maximum(cdf, 1e-10)
    mills_ratio = pdf / cdf_safe  # phi(z)/Phi(z)
    
    # Conditional expected improvement given new best
    cond_ei = (f_best - mu) + sigma_safe * mills_ratio
    
    # KG approximation: P(discovery) * E[improvement | discovery]
    # This equals EI exactly in 1D, but the structural decomposition
    # allows us to weight each component differently for better navigation
    # in the multimodal Griewank landscape.
    
    # The key structural change: instead of using EI directly, we use
    # a weighted combination that emphasizes discovery probability more
    # for flat regions (where mu is close to f_best) and value reduction
    # more for steep regions.
    
    # Weight discovery probability with a factor that increases when
    # the GP is uncertain (high sigma) relative to the current best gap
    gap = f_best - mu  # Positive when x is predicted to be better
    uncertainty_ratio = sigma_safe / (np.abs(gap) + 1e-8)
    
    # When uncertainty is high relative to the gap, discovery probability
    # is more informative. Scale the discovery component.
    discovery_weight = 1.0 + 0.5 * np.clip(uncertainty_ratio, 0, 3)
    
    # KG score: weighted combination
    # Component 1: Discovery probability (probability of finding new best)
    # Component 2: Expected value reduction (how much better it would be)
    kg_discovery = cdf * discovery_weight
    kg_value = cond_ei * cdf  # Same as EI but computed via decomposition
    
    # Combine: emphasize discovery for exploration in flat regions,
    # and value reduction for exploitation in promising regions
    # Use a smooth blend: when cdf is high (likely new best), weight value more;
    # when cdf is low (unlikely), weight discovery more for exploration.
    blend = 0.3 * cdf + 0.7  # Ranges from 0.3 to 1.0
    
    kg_score = blend * kg_value + (1 - blend) * kg_discovery * sigma_safe
    
    # Cost-aware normalization:
    # Cost ranges from ~1.0 (edges) to ~2.0 (center)
    # We want to prefer cheaper evaluations but not too aggressively
    # since the minimum is in the inner 80% (cost ~1.2-2.0)
    # Use a moderate cost penalty
    cost_penalty = 1.0 + 0.3 * (cost - 1.0)
    
    score = kg_score / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
