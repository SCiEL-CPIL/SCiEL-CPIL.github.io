import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin for 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Hybrid Acquisition: Combine Expected Improvement (EI) and Thompson Sampling (TS)
    # EI provides stable exploitation of known good regions.
    # TS provides stochastic exploration, especially useful early on.
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -10, 10)
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # 2. Compute Thompson Sampling (TS) score
    # We want to sample y ~ N(mu, sigma^2) and minimize y.
    # To balance exploration/exploitation, we can scale the variance.
    # If f_best is high (early), we want more exploration (larger std).
    # If f_best is low (late), we want exploitation (smaller std, closer to mu).
    
    # Dynamic scaling factor for TS std
    # f_best ranges from ~100 down to 0.
    # Let's use a scaling that increases exploration when f_best is large.
    # kappa = 1 + alpha * f_best
    # If f_best=100, kappa=1+alpha*100. If alpha=0.01, kappa=2.
    # If f_best=0, kappa=1.
    
    # Alternatively, simply use the posterior std directly.
    # The current program used kappa = 1 + f_best/50.
    # Let's try a milder exploration boost or just standard TS.
    # Given EI is already strong, maybe just standard TS is better?
    # Or a mix.
    
    # Let's try a weighted sum.
    # Score = w * (EI / cost) + (1-w) * (-y / cost)
    # But EI is deterministic, TS is stochastic.
    # A common approach is to use EI as the primary driver but add a small noise term
    # or use TS when uncertainty is high.
    
    # Let's stick to the structure of the current program (TS-based) but refine it,
    # or the structure of the seed (EI-based) but refine it.
    # The current program (TS) scored 1.29. The seed (EI) scored 1.0.
    # This suggests TS is performing better in this specific setup, possibly due to
    # the cost-awareness interacting better with stochastic sampling.
    
    # However, Inspiration 1 (EI) is the baseline.
    # Let's combine: Use EI as the base score, but add a Thompson Sampling exploration term
    # that is scaled by the uncertainty.
    
    # Actually, a cleaner combination is:
    # Score = (EI + lambda * sigma * Z) / cost
    # Where Z is a standard normal random variable. This is effectively
    # (f_best - mu)*Phi(z) + sigma*phi(z) + lambda*sigma*Z
    # This looks like a noisy EI.
    
    # Let's try a simpler blend:
    # EI_score = EI / cost
    # TS_score = -y / cost  where y ~ N(mu, sigma)
    
    # Let's compute both and take a convex combination.
    # w_ei = 0.7, w_ts = 0.3
    # As f_best decreases, we might want to increase w_ei (more exploitation).
    
    # f_best is a scalar.
    # Let's map f_best to a weight.
    # If f_best > 10, we are far away. w_ei = 0.5, w_ts = 0.5
    # If f_best < 1, we are close. w_ei = 0.9, w_ts = 0.1
    
    # Linear interpolation:
    # w_ei = 0.5 + 0.4 * (1 - min(f_best/10, 1.0))
    # If f_best=10, w_ei=0.5
    # If f_best=0, w_ei=0.9
    
    # Clamp f_best for the weight calculation
    f_best_for_weight = min(max(f_best, 0.0), 10.0)
    w_ei = 0.5 + 0.4 * (1.0 - f_best_for_weight / 10.0)
    w_ts = 1.0 - w_ei
    
    # Compute EI Score
    ei_scores = ei / cost
    
    # Compute TS Score
    # Sample from posterior
    y = np.random.normal(mu, sigma)
    ts_scores = -y / cost
    
    # Combine
    scores = w_ei * ei_scores + w_ts * ts_scores
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
