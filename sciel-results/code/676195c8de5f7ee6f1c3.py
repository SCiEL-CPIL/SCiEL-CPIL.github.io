import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for Bayesian Optimization.
    
    Instead of using a fixed kappa (exploration parameter), this method
    determines the required exploration level to achieve a target probability
    of improvement (e.g., 50% chance that the new observation is better than f_best).
    
    For minimization, we want P(f(x) < f_best) >= p_target.
    Assuming f(x) ~ N(mu, sigma^2), this is:
    P((f(x) - mu) / sigma < (f_best - mu) / sigma) >= p_target
    norm.cdf((f_best - mu) / sigma) >= p_target
    (f_best - mu) / sigma >= norm.ppf(p_target)
    -mu + f_best >= sigma * norm.ppf(p_target)
    mu - sigma * norm.ppf(p_target) <= f_best
    
    The "score" for selection should be high for points likely to be good.
    We can define the score as the "effective upper bound" we need to beat.
    Actually, a simpler and robust formulation for adaptive UCB is:
    
    We want to select the point that maximizes the probability of improvement,
    or equivalently, we can use a UCB variant where the threshold is set
    based on the distribution of the means.
    
    A common adaptive UCB (like in the "UCB with adaptive threshold" literature)
    sets the threshold tau such that we explore points where mu - kappa*sigma is low.
    To make it adaptive, we can set kappa such that we are targeting a specific
    quantile of the expected improvement or simply use the posterior mean's
    distribution to set the exploration bonus.
    
    Let's use the formulation:
    score = mu - kappa * sigma
    
    Where kappa is chosen such that the probability of improvement is constant.
    Alternatively, a very effective simple strategy for short horizons is
    "Expected Hypervolume Improvement" (EHVI) or just "Expected Improvement" (EI).
    
    However, the prompt asks for a strategy MATERIALLY DIFFERENT from the current
    fixed-kappa UCB. The current one uses kappa=1.0.
    
    Let's implement an Adaptive UCB where the exploration term is scaled by the
    standard deviation of the posterior mean across all candidates, or more simply,
    we use a quantile-based approach.
    
    Strategy: 
    1. Compute the standard deviation of the posterior means (mu).
    2. Set the exploration bonus to be a multiple of this standard deviation? No, that's not quite right.
    
    Let's try the "Probability of Improvement" (PI) or a smoothed version of it.
    PI(x) = P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    This is bounded between 0 and 1. It is smooth.
    However, PI can be too greedy if sigma is very small.
    
    A better "different" strategy is **Expected Improvement (EI)**.
    EI(x) = E[max(f_best - f(x), 0)]
    EI(x) = (f_best - mu) * norm.cdf((f_best - mu) / sigma) + sigma * norm.pdf((f_best - mu) / sigma)
    
    EI is a standard, robust acquisition function that is materially different from
    a linear UCB score. It naturally balances exploration and exploitation without
    needing a manually tuned kappa.
    
    Given the short horizon (30 evals) and 1D domain, EI is often superior to UCB
    because it focuses on the actual expected gain, not just the upper bound.
    
    Let's implement EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF of the standard normal
    # phi(z) is the PDF of the standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # We want to maximize EI, so the score is EI itself.
    # The evaluator picks the candidate with the highest score.
    return ei
