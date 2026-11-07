"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using quantile-based feasibility gate
    combined with adaptive UCB blending.
    
    Structural change: Instead of smooth trade-off between EI and cost,
    we use a hard feasibility gate based on the 25th percentile of cost,
    and within the feasible set, we blend EI and UCB based on local
    uncertainty to balance exploitation and exploration.
    """
    n = len(mu)
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we want low values
    # UCB = mu - kappa * sigma (lower is better)
    # But since we're maximizing the acquisition, we use:
    # acquisition_ucb = -(mu - kappa*sigma) = -mu + kappa*sigma
    # Actually, for minimization with UCB, we pick point with lowest mu - kappa*sigma
    # So the "score" to maximize is: -mu + kappa*sigma
    # But EI already incorporates uncertainty. Let's think differently.
    
    # For minimization:
    # EI favors points where mu < f_best (expected improvement)
    # UCB favors points with high uncertainty (high sigma) and low mu
    
    # Adaptive kappa based on fraction of explored space
    # Early: more exploration (higher kappa), Late: more exploitation (lower kappa)
    # We don't have iteration count, so use uncertainty level as proxy
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    if max_sigma > 1e-9:
        uncertainty_ratio = mean_sigma / max_sigma
    else:
        uncertainty_ratio = 0.0
    
    # kappa decreases as uncertainty decreases (more exploitation later)
    kappa = 0.5 + 2.0 * uncertainty_ratio  # ranges from 0.5 to 2.5
    
    # UCB score for minimization: we want to maximize -(mu - kappa*sigma)
    # which is equivalent to maximizing kappa*sigma - mu
    ucb_score = kappa * sigma - mu
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(ucb_score)
    if ucb_max > 1e-9:
        ucb_norm = ucb_score / ucb_max
    else:
        ucb_norm = ucb_score
    
    # Blend EI and UCB based on uncertainty ratio
    # High uncertainty: more UCB, Low uncertainty: more EI
    alpha = 0.5 + 0.5 * uncertainty_ratio  # weight for UCB
    blended = (1 - alpha) * ei_norm + alpha * ucb_norm
    
    # Quantile-based feasibility gate
    # Use 25th percentile as threshold - candidates with cost below this are "feasible"
    cost_q25 = np.percentile(cost, 25)
    cost_q75 = np.percentile(cost, 75)
    
    # Smooth gate: 1.0 for cost <= q25, 0.0 for cost >= q75, linear in between
    if cost_q75 > cost_q25:
        gate = np.clip((cost_q75 - cost) / (cost_q75 - cost_q25 + 1e-9), 0.0, 1.0)
    else:
        gate = np.where(cost <= cost_q25, 1.0, 0.5)
    
    # Apply gate to blended score
    score = blended * gate
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
