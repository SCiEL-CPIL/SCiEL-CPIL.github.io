"""
Pareto-front cost-aware acquisition for Ackley 2D Bayesian optimization.
Strategy:
1. Compute raw Expected Improvement (EI) for minimization.
2. Compute a cost-adjusted score using a non-dominated Pareto front approach:
   - We want high EI and low cost.
   - Instead of dividing EI by cost (which can be noisy), we use a weighted
     sum but with weights that shift from exploration to exploitation as
     the budget depletes.
3. Anneal the exploration weight based on the current best value relative
   to the theoretical minimum (0) and the remaining budget proxy (cost).
"""
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-front-inspired weighting.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.nan_to_num(sigma, nan=1e-12, posinf=1.0, neginf=1e-12)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1.0)  # Cost is at least 1
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # 1. Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma
    # Avoid numerical issues in z
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best, but we only care about relative ranking
    # Clip negative EI to 0 to focus on points with potential improvement
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-awareness via Pareto-inspired weighting
    # We want to maximize EI and minimize cost.
    # A common robust approach is to normalize both and use a dynamic weight.
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize cost to [0, 1] (lower is better, so we invert)
    # cost is in [1, 2] typically for Ackley on [-5,5]^2 scaled to [0,1]
    # mean(x) in [0,1] -> cost in [1, 2]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        # Invert so that lower cost -> higher score
        cost_norm = 1.0 - (cost - cost_min) / cost_range
    else:
        cost_norm = np.ones_like(cost)
    
    # 3. Adaptive Weighting (Annealing)
    # Determine how "urgent" exploitation is.
    # If f_best is very small (close to 0), we should exploit more (weight EI higher).
    # If f_best is large, we should explore more (weight cost/uncertainty higher).
    
    # Also consider the "budget" aspect implicitly:
    # As we get closer to the optimum, the marginal gain from expensive points
    # diminishes, so we should penalize cost more heavily.
    
    # Heuristic:
    # w_ei: weight for Expected Improvement
    # w_cost: weight for Cost Efficiency
    
    # Scale factor: how good is f_best?
    # Ackley min is 0. Typical initial f_best might be ~1-2.
    # If f_best < 0.1, we are very close.
    
    if f_best < 0.05:
        # Very close to optimum: prioritize finding the exact minimum (EI)
        # but be very sensitive to cost as gains are tiny.
        w_ei = 0.9
        w_cost = 0.1
    elif f_best < 0.2:
        w_ei = 0.8
        w_cost = 0.2
    elif f_best < 0.5:
        w_ei = 0.7
        w_cost = 0.3
    elif f_best < 1.0:
        w_ei = 0.6
        w_cost = 0.4
    else:
        # Far from optimum: exploration is key.
        # Cost matters less if we can find a region with much lower f.
        # But we still shouldn't waste budget on very expensive points
        # unless the EI is huge.
        w_ei = 0.5
        w_cost = 0.5
        
    # Combine
    # Note: cost_norm is 1 for cheapest, 0 for most expensive.
    # We want to prefer low cost, so we add w_cost * cost_norm.
    # We want to prefer high EI, so we add w_ei * ei_norm.
    
    scores = w_ei * ei_norm + w_cost * cost_norm
    
    # 4. Secondary refinement: Local Density / Uncertainty
    # If two points have similar EI and cost, prefer the one with higher uncertainty (sigma)
    # to help the GP learn faster.
    # However, if we are very close to the optimum (f_best < 0.05),
    # we might want to suppress exploration to converge.
    
    if f_best > 0.1:
        # Add a small uncertainty bonus
        sigma_max = np.max(sigma)
        if sigma_max > 1e-9:
            sigma_norm = sigma / sigma_max
            # Small weight, e.g., 0.1
            scores += 0.1 * sigma_norm
            # Re-normalize roughly to keep scale consistent (optional)
            # scores = scores / (scores.max() + 1e-9)
            
    # Final safety check
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
