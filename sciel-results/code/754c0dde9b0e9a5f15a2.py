# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # Structural change: Use a knowledge-gradient-style expected improvement
    # that models the probability of being the global best, combined with
    # cost-awareness. This differs from standard EI by considering the
    # relative position among all candidates.

    # First, compute standard EI for minimization
    z = (f_best - mu_safe) / sigma_safe
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Knowledge gradient style: probability that candidate i is the global best
    # among all candidates. This is approximated by considering the CDF of
    # each candidate's posterior relative to the current best.
    # P(x_i is best) ≈ P(f(x_i) < f_best) = norm.cdf((f_best - mu_i) / sigma_i)
    # But we want to account for the fact that other candidates might also be good.
    # A simpler approximation: weight EI by the probability of improvement
    pi = norm.cdf(z)

    # Adaptive LCB with dynamic exploration parameter
    # beta scales with relative uncertainty: when sigma is large relative to
    # the range of mu, explore more
    mu_range = np.ptp(mu_safe) if n > 1 else 1.0
    if mu_range < 1e-9:
        mu_range = 1.0
    sigma_mean = np.mean(sigma_safe)
    relative_uncertainty = sigma_mean / (mu_range + sigma_mean + 1e-9)
    
    # beta: higher when uncertainty is high relative to signal
    beta = 0.5 + 2.0 * relative_uncertainty

    # LCB for minimization: lower is better, so we negate for maximization
    lcb = -(mu_safe + beta * sigma_safe)

    # Cost adjustment: divide by cost to get "improvement per unit cost"
    cost_adj_ei = ei / cost_safe
    cost_adj_lcb = lcb / cost_safe
    cost_adj_pi = pi / cost_safe

    # Blend: primarily use cost-adjusted EI, with a component from LCB
    # for exploration, and a small component from probability of improvement
    # for robustness
    
    # Normalize each component to comparable scale
    ei_max = np.max(cost_adj_ei) if n > 0 else 1.0
    lcb_max = np.max(cost_adj_lcb) if n > 0 else 1.0
    pi_max = np.max(cost_adj_pi) if n > 0 else 1.0
    
    if ei_max < 1e-9:
        ei_max = 1.0
    if lcb_max < 1e-9:
        lcb_max = 1.0
    if pi_max < 1e-9:
        pi_max = 1.0

    # Weighted blend: EI dominates, LCB provides exploration, PI provides stability
    w_ei = 0.6
    w_lcb = 0.3
    w_pi = 0.1

    scores = (w_ei * cost_adj_ei / ei_max + 
              w_lcb * cost_adj_lcb / lcb_max + 
              w_pi * cost_adj_pi / pi_max)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    # If time permits, try an alternative: pure cost-adjusted LCB
    # and compare which one has lower mean mu in top-k
    if n >= 100 and time.time() < deadline - 10.0:
        alt_scores = cost_adj_lcb
        alt_scores = np.where(np.isfinite(alt_scores), alt_scores, 0.0)
        
        top_k = min(10, n // 10)
        if top_k >= 1:
            idx_current = np.argsort(scores)[-top_k:]
            idx_alt = np.argsort(alt_scores)[-top_k:]
            
            mean_mu_current = np.mean(mu_safe[idx_current])
            mean_mu_alt = np.mean(mu_safe[idx_alt])
            
            # Prefer the one with lower mean mu in top-k
            if mean_mu_alt < mean_mu_current - 1e-6:
                scores = alt_scores

    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
