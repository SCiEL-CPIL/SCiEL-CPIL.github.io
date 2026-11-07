"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG estimates the expected reduction in the global minimum value
    by approximating the maximum of the predictive distribution over
    the candidate set. This prioritizes points most likely to yield
    a new global best rather than just high uncertainty or immediate improvement.
    
    For each candidate i, KG_i = E[max(0, f_best - Y_i)] where Y_i ~ N(mu_i, sigma_i)
    But a more accurate KG considers the expected improvement over the minimum
    of all future observations. Here we use a simplified but effective approximation:
    
    KG_i ≈ E[max(0, f_best - Y_i)] - E[max(0, f_best - Y_j)] for the "best" other point
    
    A practical approximation: compute the expected improvement of each point
    relative to f_best, but weight by the probability that this point is the
    minimum among all candidates. This is expensive, so we use a surrogate.
    
    We use the following approximation:
    1. Compute EI for each point: EI_i = E[max(0, f_best - Y_i)]
    2. Estimate the probability that point i is the minimum: P(Y_i < Y_j for all j≠i)
       Approximate by: P(min_j Y_j < f_best) weighted by each point's contribution
    
    A simpler and fast approximation that works well in practice:
    KG_i ≈ EI_i * (1 - exp(-sigma_i^2 / (2 * f_best^2)))
    This boosts points with high uncertainty relative to the current best.
    
    Actually, let's use a well-known fast KG approximation:
    For minimization, KG_i = E[max(0, f_best - Y_i)] where we account for
    the fact that we're looking for the global minimum.
    
    The standard EI for minimization:
    EI_i = (f_best - mu_i) * Phi((f_best - mu_i)/sigma_i) + sigma_i * phi((f_best - mu_i)/sigma_i)
    
    For KG, we want the expected reduction in the minimum. A good approximation
    is to compute EI but with a correction for the number of candidates.
    
    Let's use: KG_i = EI_i * correction_factor
    where correction_factor accounts for the fact that we're selecting one point
    from many.
    
    Actually, the most practical and effective approach for this problem size
    (2000 candidates) is to use a simplified KG:
    
    KG_i = E[max(0, f_best - Y_i)] * P(Y_i is the minimum)
    
    P(Y_i is the minimum) can be approximated by:
    P(Y_i < min_{j≠i} Y_j)
    
    For a fast approximation, we can use:
    - Compute the "effective" number of points that could beat f_best
    - Weight EI by the inverse of this count
    
    Let's implement a practical KG:
    1. Compute EI for all points
    2. Compute the probability that each point is the global minimum
       using a softmax-like weighting based on mu and sigma
    3. KG_i = EI_i * prob_i
    
    For efficiency, we'll use:
    prob_i ∝ exp(-mu_i / (sigma_i + eps)) * sigma_i
    This gives higher probability to points with low mu and moderate sigma.
    
    Actually, let's use the standard KG approximation for Gaussian processes:
    
    KG_i = E[max(0, f_best - Y_i)] - E[max(0, f_best - Y_{-i}^{min})]
    
    where Y_{-i}^{min} is the minimum of all other points. This is hard to compute
    exactly, but we can approximate it.
    
    A well-known fast approximation:
    KG_i ≈ EI_i * (1 - (1 - Phi((f_best - mu_i)/sigma_i))^N)
    where N is the number of candidates. This accounts for the probability
    that at least one other point beats f_best.
    
    Let's implement this:
    """
    # Track time for safety
    t0 = time.time()
    deadline = 325.0  # Leave 5s margin before 330s kill
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardize for numerical stability
    d = (f_best - mu) / sigma_safe
    
    # Compute EI for minimization
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    EI = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Ensure EI is non-negative (numerical issues)
    EI = np.maximum(EI, 0.0)
    
    # KG approximation:
    # The probability that point i is the unique minimum below f_best
    # is approximately: Phi(d_i) * product_{j≠i}(1 - Phi(d_j))
    # This is expensive, so we use an approximation.
    
    # Approximation: KG_i = EI_i * w_i
    # where w_i = 1 / (1 + sum_{j≠i} exp(-(d_j - d_i)))
    # This is a softmax-like weighting.
    
    # For efficiency with 2000 points, use a simpler approximation:
    # The expected number of points below f_best is sum_j Phi(d_j)
    # The probability that point i is the minimum among those below f_best
    # is approximately Phi(d_i) / sum_j Phi(d_j)
    
    sum_Phi = np.sum(Phi_d)
    if sum_Phi < 1e-12:
        sum_Phi = 1e-12
    
    # Weight: probability that this point is the most likely new minimum
    weights = Phi_d / sum_Phi
    
    # KG score
    KG = EI * weights
    
    # Check time
    if time.time() - t0 > deadline:
        # Return best available
        KG = np.nan_to_num(KG, nan=0.0, posinf=0.0, neginf=0.0)
        return KG
    
    # Ensure finite scores
    KG = np.nan_to_num(KG, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add a small exploration bonus for very early stages
    # If f_best is still high (poor), explore more
    exploration_bonus = 0.1 * sigma_safe / (abs(f_best) + 1e-12)
    KG = KG + exploration_bonus
    
    # Final safety check
    KG = np.nan_to_num(KG, nan=0.0, posinf=0.0, neginf=0.0)
    
    return KG
# EVOLVE-BLOCK-END
