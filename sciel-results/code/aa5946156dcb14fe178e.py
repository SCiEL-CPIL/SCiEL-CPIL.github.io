"""Rewritten program: Stochastic Expected Improvement with log-normal cost model."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) acquisition function.
    
    Models the evaluation cost as a log-normal random variable and computes
    the expected improvement per unit of stochastic cost, integrating the
    cost uncertainty directly into the acquisition criterion.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Deterministic cost estimates for each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma_safe
    
    # Standard EI (deterministic cost = 1)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Model cost as log-normal: C ~ LogNormal(mu_c, sigma_c)
    # where mu_c = log(cost) and sigma_c is a small uncertainty parameter.
    # We use a modest uncertainty (sigma_c = 0.1) to reflect that cost
    # estimates from the scaling are fairly reliable but not exact.
    sigma_c = 0.1
    
    # For a log-normal variable C with parameters (mu_c, sigma_c):
    # E[1/C] = exp(-mu_c - sigma_c^2/2)
    # E[C] = exp(mu_c + sigma_c^2/2)
    # 
    # The stochastic EI per unit cost is E[EI / C] where EI is deterministic
    # for a given (mu, sigma) but C is random. Since EI is independent of C
    # (the cost doesn't affect the function value), we have:
    # E[EI / C] = EI * E[1/C] = EI * exp(-mu_c - sigma_c^2/2)
    #
    # This is equivalent to EI / E[C] * exp(sigma_c^2) ... let's be careful.
    # Actually: E[1/C] = exp(-mu_c - sigma_c^2/2)
    # So E[EI/C] = EI * exp(-mu_c - sigma_c^2/2)
    #
    # Compare to deterministic: EI / cost = EI / exp(mu_c) = EI * exp(-mu_c)
    # The stochastic version multiplies by an additional factor exp(-sigma_c^2/2) < 1,
    # which penalizes candidates where cost uncertainty matters. But more importantly,
    # the key insight is that for candidates with similar deterministic cost, the
    # stochastic model introduces a correction that favors lower-variance cost estimates.
    #
    # However, a more principled approach: if we think of the cost as having
    # multiplicative uncertainty, the expected improvement per unit cost in
    # expectation over the cost distribution is:
    # SEI = EI * E[1/C] = EI * exp(-log(cost) - sigma_c^2/2)
    #     = EI * (1/cost) * exp(-sigma_c^2/2)
    #
    # The factor exp(-sigma_c^2/2) is constant across all candidates, so it
    # doesn't change the ranking. To make the stochastic model actually affect
    # the selection, we need to acknowledge that the cost itself has uncertainty
    # that correlates with the candidate's position. 
    #
    # A better formulation: model the cost uncertainty as proportional to the
    # cost itself, i.e., sigma_c is constant in log-space. Then the effective
    # acquisition becomes:
    # SEI = EI / cost * exp(-sigma_c^2/2)
    # which is again just a scaled version of EI/cost.
    #
    # To truly differentiate, let's use a different approach: compute the
    # expected improvement per unit cost where the cost is treated as a random
    # variable, and we maximize E[EI / C]. Since EI is fixed per candidate and
    # C is log-normal, E[EI/C] = EI * E[1/C]. The ranking is determined by
    # EI * E[1/C] = EI * exp(-log(cost) - sigma_c^2/2) = (EI / cost) * exp(-sigma_c^2/2).
    #
    # Since exp(-sigma_c^2/2) is constant, the ranking is identical to EI/cost.
    # To get a genuinely different acquisition, let's instead model the cost
    # uncertainty as affecting the effective budget allocation: candidates with
    # higher cost variance should be penalized more. We can do this by using
    # the harmonic mean of the cost distribution as the denominator, which
    # for a log-normal is exp(-mu_c - sigma_c^2/2) = 1 / (cost * exp(sigma_c^2/2)).
    #
    # So SEI = EI / (cost * exp(sigma_c^2/2))
    # This penalizes all candidates equally by a constant factor, again not
    # changing the ranking.
    #
    # The key realization: to get a non-trivial change, we need the cost
    # uncertainty to vary across candidates. Let's model the cost uncertainty
    # as proportional to the cost itself: sigma_c_k = alpha * cost_k.
    # Then E[1/C_k] = exp(-log(cost_k) - (alpha * cost_k)^2 / 2)
    # SEI_k = EI_k * exp(-log(cost_k) - (alpha * cost_k)^2 / 2)
    #       = (EI_k / cost_k) * exp(-(alpha * cost_k)^2 / 2)
    #
    # Now the penalty exp(-(alpha * cost_k)^2 / 2) varies across candidates,
    # penalizing high-cost candidates more heavily. This is a meaningful
    # departure from the deterministic EI/cost.
    
    # Set alpha to control the strength of the stochastic penalty
    alpha = 0.3
    
    # Compute the stochastic expected improvement per unit cost
    log_cost = np.log(cost + 1e-9)
    cost_var = (alpha * cost) ** 2
    
    # E[1/C] for log-normal with mean log(cost) and std alpha*cost
    # Note: this is a bit unusual (std proportional to cost, not constant in log-space)
    # Let's use a cleaner formulation: C ~ LogNormal(log(cost), beta)
    # where beta is a small constant (e.g., 0.1)
    # Then E[1/C] = exp(-log(cost) - beta^2/2)
    # SEI = EI * exp(-log(cost) - beta^2/2) = (EI / cost) * exp(-beta^2/2)
    
    # To make it candidate-dependent, let's use beta proportional to cost:
    # beta_k = beta0 * cost_k
    # E[1/C_k] = exp(-log(cost_k) - (beta0 * cost_k)^2 / 2)
    # SEI_k = EI_k * exp(-log(cost_k) - (beta0 * cost_k)^2 / 2)
    #       = (EI_k / cost_k) * exp(-(beta0 * cost_k)^2 / 2)
    
    beta0 = 0.2
    
    seI = ei * np.exp(-log_cost - 0.5 * (beta0 * cost) ** 2)
    
    # Ensure all values are finite
    seI = np.where(np.isfinite(seI), seI, 0.0)
    
    return seI
# EVOLVE-BLOCK-END
