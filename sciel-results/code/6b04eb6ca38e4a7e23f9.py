"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) with log-normal cost model.
    
    Instead of a deterministic ratio EI/cost, we model the cost C as a
    log-normal random variable with parameters derived from the observed
    deterministic cost value. We then compute the expected improvement
    per unit of stochastic cost: E[max(f_best - f, 0) / C].
    
    For a log-normal C ~ LogNormal(mu_c, sigma_c), we have:
    E[1/C] = exp(mu_c + sigma_c^2 / 2)
    
    However, a more principled approach for cost-aware BO is to treat the
    cost as a random variable and compute:
    SEI = E[ max(f_best - f(x), 0) * (1/C) ]
    
    If f(x) and C are independent, this factors as:
    SEI = EI(x) * E[1/C]
    
    Since E[1/C] is a constant multiplier for all x, it doesn't change the
    ranking. To make it meaningful, we need to incorporate cost uncertainty
    in a way that affects the relative ranking.
    
    A better structural change: Use a two-stage acquisition where we
    explicitly model the variance of the cost. Points with higher cost
    uncertainty (higher variance in the log-normal model) are penalized
    more heavily because the expected cost is higher.
    
    Let's use a different structural idea: Pareto-efficient selection via
    a multi-objective view where we have two objectives: minimize f(x) and
    minimize cost. We can use a hypervolume-based or epsilon-constraint
    approach.
    
    Actually, let's stick to a cleaner probabilistic formulation. The key
    insight is that dividing by cost is a heuristic. A more principled
    approach is to use the Expected Improvement per unit cost, but where
    the cost itself has uncertainty. If we model cost as log-normal with
    mean = observed_cost and coefficient of variation (CV) = 0.1 (small
    uncertainty), then:
    
    E[1/C] = exp(mu_c + sigma_c^2/2) where mu_c = log(cost) - sigma_c^2/2
    and sigma_c = CV.
    
    This still gives a constant multiplier. So the ranking doesn't change.
    
    Let's try a different structural change: Use a Bayesian approach where
    we model the cost as a random variable with a distribution that depends
    on the input x. But we don't have x.
    
    Given the constraints, let's combine the EI/cost approach with a
    robustness term. We'll compute EI/cost but add a penalty for high cost
    variance. Since we don't have variance, we'll use a heuristic: assume
    cost variance is proportional to cost^2 (constant CV). Then the
    effective cost is the mean plus a penalty for variance.
    
    Let's try: effective_cost = cost * (1 + penalty * CV^2)
    where CV = 0.1 and penalty = 1.0. This gives:
    effective_cost = cost * 1.01
    
    This is still a constant multiplier.
    
    I think the best structural change is to use a different acquisition
    function family. Let's use Upper Confidence Bound (UCB) with a
    cost-aware beta. The idea is that the exploration bonus should be
    scaled by the inverse cost, so we explore more in cheap regions.
    
    UCB_cost = -mu + beta * sigma / cost
    
    This is different from EI/cost because the exploration term is also
    divided by cost, not just the exploitation term.
    
    Let's implement this.
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    # Safety: avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Compute cost-aware EI: EI / cost
    ei_cost = ei / cost_safe
    
    # Compute cost-aware UCB: -mu + beta * sigma / cost
    # Choose beta to balance exploration and exploitation
    # In standard UCB, beta is often around 2-4. For cost-aware, we use
    # a smaller beta since the exploration is already discounted by cost.
    beta = 1.5
    ucb_cost = -mu + beta * sigma_safe / cost_safe
    
    # Combine EI/cost and UCB/cost with equal weight
    # This gives a hybrid that benefits from both exploitation and exploration
    score = 0.5 * ei_cost + 0.5 * ucb_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
