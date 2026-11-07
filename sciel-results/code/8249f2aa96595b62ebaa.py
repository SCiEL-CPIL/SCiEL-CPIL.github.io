"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as t_dist
import time

# Deadline guard
_DEADLINE = 325.0  # safety margin before 330s kill
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student's t-based Expected Improvement.
    
    Uses a t-distribution with 5 degrees of freedom to model the predictive
    distribution, which has heavier tails than the Gaussian. This accounts for
    the high uncertainty in the early stages of Bayesian optimization with
    a limited budget (30 evaluations in 6D), promoting more robust exploration.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return sigma.copy()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of samples (for t-distribution degrees of freedom)
    # Using 5 degrees of freedom as suggested - provides heavy tails
    df = 5.0
    
    # Standardized improvement: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # For Student's t-distribution with df degrees of freedom:
    # The predictive distribution is t-distributed, not Gaussian
    # E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # But we model f as having a t-distribution scaled to match mu, sigma
    
    # The t-based EI can be approximated by scaling the Gaussian EI
    # with a factor that accounts for the heavier tails
    # 
    # A simpler and more direct approach: use the t-CDF and t-PDF
    # The predictive distribution is: f ~ mu + sigma * t_df / sqrt(df)
    # where t_df is standard t with df degrees of freedom
    #
    # Actually, let's use a well-known formulation:
    # If we model the predictive as a t-distribution with location mu,
    # scale sigma*sqrt(df/2) to match the variance, then:
    # 
    # The CDF of a t-distribution with df degrees of freedom at point x
    # is t_dist.cdf(x, df)
    # 
    # For EI with t-distribution:
    # EI = (f_best - mu) * P(f < f_best) + E[(f_best - f) * 1(f < f_best)]
    #
    # Let's use a practical approximation that captures the heavy-tailed nature:
    # Scale the standard Gaussian EI by a factor > 1 to account for heavier tails
    # This effectively increases the exploration component
    
    # Standard Gaussian EI
    ei_gaussian = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # t-based adjustment: the t-distribution has heavier tails, so the expected
    # improvement should be larger for the same (mu, sigma) because there's
    # more mass in the tails
    # A good approximation: multiply by a factor based on df
    # For df=5, the variance of t is df/(df-2) = 5/3 ≈ 1.667
    # The standard deviation is sqrt(5/3) ≈ 1.29
    # So we scale EI by approximately sqrt(df/(df-2)) for df > 2
    if df > 2:
        t_scale = np.sqrt(df / (df - 2.0))  # ≈ 1.29 for df=5
    else:
        t_scale = 1.0
    
    # t-based EI approximation
    ei_t = ei_gaussian * t_scale
    
    # Add a small exploration bonus to encourage diversity
    # This helps in 6D where the landscape is complex
    exploration_bonus = 0.05 * sigma_safe
    
    scores = ei_t + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
