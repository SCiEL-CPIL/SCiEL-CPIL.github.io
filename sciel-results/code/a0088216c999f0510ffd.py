"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that blends a global target strategy (distance to 0,0)
    with local uncertainty exploitation (UCB).
    
    The Rosenbrock function f(x) = (1-x1)^2 + 100(x2-x1^2)^2 has a global minimum
    at (0,0) with f(0,0)=0. The domain is [-2, 2]^2.
    We reconstruct approximate coordinates from the cost signal to bias search
    towards the center, but we primarily rely on the GP posterior to identify
    promising regions.
    
    Strategy:
    1. Compute UCB for local exploration.
    2. Compute a "global pull" towards the known minimum (0,0).
    3. Combine them.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Handle edge cases where sigma is 0 or negative
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # 1. UCB: Uncertainty-weighted improvement
    # Using a beta parameter to balance exploration.
    # Since we have limited budget (cost < 30), and initial points are random,
    # we should be relatively aggressive in exploration early on, 
    # but the GP will guide us.
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # 2. Global Target Strategy
    # The minimum of Rosenbrock is at x = (0, 0).
    # The cost function is cost = 1 + mean(x_scaled), where x_scaled = (x+2)/4.
    # This implies cost is higher for points with larger x values.
    # However, we don't have x directly. We have mu, sigma, cost.
    # 
    # Insight: The GP provides mu (predicted f) and sigma (predicted std of f).
    # If we trust the GP, we should pick low mu.
    # But the GP might be wrong in unexplored regions.
    # 
    # A different angle: The prompt mentions "rosenbrock 2d".
    # The minimizer is moved to a uniform point in the inner 80% of each range.
    # Wait, the prompt says: "minimizes rosenbrock 2d on [-2, 2]^2 (min 0). 
    # Each function runs on five seeded instances f_k(x) = f(x - s_k) whose minimizer 
    # is moved to a uniform point in the inner 80% of each range".
    # 
    # This means the actual minimum is NOT at (0,0) in the original coordinates!
    # It is at s_k, where s_k is a uniform point in [-1.6, 1.6]^2 (inner 80% of [-2,2]).
    # 
    # Therefore, targeting (0,0) is WRONG. The minimum is shifted.
    # We cannot assume the minimum is at (0,0). We must rely on the GP.
    #
    # However, the cost structure is still useful.
    # cost = 1 + mean((x+2)/4) = 1 + (mean(x) + 2)/4.
    # Lower cost corresponds to lower mean(x).
    # This doesn't directly help locate the minimum, but it tells us about the location.
    #
    # Let's stick to a robust Bayesian optimization acquisition.
    # The current seed uses EI / cost.
    # EI is good for exploitation. UCB is good for exploration.
    # 
    # Let's try a hybrid: 
    # Score = -mu + kappa * sigma
    # This is standard UCB.
    # 
    # Why did the seed (EI/cost) score 1.0?
    # EI/cost penalizes high-cost evaluations.
    # If the minimum is in a high-cost region (high x), EI/cost will avoid it.
    # If the minimum is in a low-cost region (low x), EI/cost will favor it.
    # Since the shift s_k is uniform, the minimum can be anywhere.
    # Penalizing cost might be harmful if the true minimum happens to be at high cost.
    # 
    # A better strategy: Pure UCB or EI without cost normalization, 
    # or a very mild cost penalty.
    # 
    # Let's try pure UCB with a tuned beta.
    # Or Expected Improvement (EI) without cost division.
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate UCB
    # We need to choose beta. 
    # A common choice is beta = sqrt(2 * log(t)) where t is the number of iterations.
    # We don't have t directly, but we have cost.
    # Initial cost is ~1 + mean(x_scaled). 
    # Let's estimate iterations from cost? No, cost is per point.
    # 
    # Let's use a fixed beta that balances exploration/exploitation well for 2D.
    # beta = 1.0 to 3.0 is typical.
    # Let's try a dynamic beta based on sigma magnitude? No, keep it simple.
    
    # Alternative: Knowledge Gradient? Too complex.
    
    # Let's try a blend of EI and UCB.
    # Normalize EI and UCB?
    
    # Actually, the "inspiration 1" failed (0.22) because it tried to target (0,0) 
    # which is incorrect due to the shift s_k.
    # 
    # The "inspiration 2" (score 0.06) is unknown but likely poor.
    # 
    # The seed (1.0) uses EI/cost.
    # 
    # Hypothesis: The cost penalty in the seed is suboptimal because it biases
    # against high-cost regions, but the minimum could be there.
    # Removing the cost penalty (using pure EI or UCB) might allow finding
    # the minimum regardless of its cost.
    
    # Let's try UCB with beta=2.0.
    score = mu + 2.0 * sigma_safe
    
    # To be safe and ensure higher values are better:
    # UCB is higher for high uncertainty or high mu?
    # Wait, for minimization:
    # We want to find LOW f.
    # GP predicts mu (low is good) and sigma (high is uncertain).
    # Standard UCB for minimization is: -mu + beta * sigma.
    # Or: mu - beta * sigma? No.
    # If mu is low, we want high score. So -mu.
    # If sigma is high, we want high score. So + beta * sigma.
    # Score = -mu + beta * sigma.
    
    score = -mu + 2.0 * sigma_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
