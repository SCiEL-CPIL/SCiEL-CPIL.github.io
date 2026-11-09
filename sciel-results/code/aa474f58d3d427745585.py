import numpy as np
from scipy.stats import norm
import time

# Deadline guard constants
_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization acquisition function.
    
    Instead of EI or UCB, we directly approximate the expected reduction in simple regret.
    Simple regret is defined as f_best - min(f(x)).
    We approximate E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    This leads to a score proportional to the expected regret reduction, normalized by cost.
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        cost_arr = np.asarray(cost, dtype=np.float64)
        safe_cost = np.where(cost_arr < 1e-9, 1e-9, cost_arr)
        return 1.0 / safe_cost
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute the regret reduction term
    # E[max(0, f_best - f(x))] is the standard EI, but we want a simpler proxy for regret minimization
    # The directive suggests: E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    # Regret reduction ≈ E[f_best - min(f(x), f_best)] = 0.5 * max(0, f_best - mu - 0.5*sigma)
    
    # Let's refine this. 
    # If we assume f(x) ~ N(mu, sigma^2), then E[max(0, f_best - f(x))] = EI.
    # The simple regret is R = f_best - min(f(x)).
    # We want to maximize E[R] = E[f_best - min(f(x))] = f_best - E[min(f(x))].
    # E[min(f(x), f_best)] is not exactly what we want if f(x) can be < f_best.
    # Actually, simple regret is f_best - f(x) if f(x) < f_best, else 0.
    # So E[Regret Reduction] = E[max(0, f_best - f(x))] = EI.
    
    # However, the directive specifically asks to try:
    # "Bayesian Regret Minimization objective by approximating the expected simple regret as 
    #  E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)"
    
    # Let's implement exactly what is suggested in the directive.
    # The term "expected simple regret" here seems to be a proxy for the potential improvement.
    # Let's compute the term T = 0.5 * max(0, f_best - mu - 0.5*sigma)
    # This is a simpler, more aggressive exploitation term than EI.
    # It ignores the tail probability (CDF) and just looks at the mean shift relative to a half-sigma threshold.
    
    # To balance exploration and exploitation, we might need to add a small exploration term,
    # or rely on the cost normalization and the GP's sigma to guide us.
    # But the directive says "Make a STRUCTURAL change... different design...".
    # So let's stick to the proposed formula primarily.
    
    # Let's compute the base score based on the proposed approximation.
    # score_raw = 0.5 * np.maximum(0, f_best - mu_safe - 0.5 * safe_sigma)
    
    # However, pure exploitation might get stuck. 
    # The "Bayesian Regret Minimization" usually implies minimizing E[f(x) - f*], but here we are minimizing f.
    # Let's look at the term again: f_best - 0.5 * max(0, f_best - mu - 0.5*sigma).
    # This is the approximated E[min(f(x), f_best)].
    # We want to MINIMIZE this value (since lower f is better).
    # So the acquisition function should be NEGATIVE of this, or we maximize ( - E[min(f(x), f_best)] ).
    # Let's maximize: - (f_best - 0.5 * max(0, f_best - mu - 0.5*sigma))
    # = 0.5 * max(0, f_best - mu - 0.5*sigma) - f_best
    
    # Since f_best is constant for all candidates, maximizing the score is equivalent to maximizing:
    # 0.5 * max(0, f_best - mu - 0.5*sigma)
    
    # This is very similar to EI but without the probabilistic weighting (CDF/PDF).
    # It is a "hard" exploitation strategy.
    
    # To ensure we still explore when mu is high (far from f_best), the term becomes 0.
    # This means all points with mu > f_best - 0.5*sigma get score 0 (before cost normalization).
    # This might be too aggressive.
    
    # Let's add a small exploration term to prevent total stagnation in high-uncertainty regions.
    # Exploration term: sigma * kappa
    # But the directive says "different design or modelling family".
    # Let's try to use the proposed formula as the primary driver, but scale it.
    
    # Proposed Score:
    # S = (0.5 * max(0, f_best - mu - 0.5*sigma)) / cost
    
    # Let's compute this.
    term = np.maximum(0, f_best - mu_safe - 0.5 * safe_sigma)
    base_score = 0.5 * term
    
    # To avoid getting stuck if the GP is confident but wrong (mu < f_best but actual f is high),
    # we might want to retain some influence of sigma.
    # However, the term already includes sigma in the threshold.
    
    # Let's normalize by cost.
    scores = base_score / safe_cost
    
    # If all scores are 0 (e.g., early on if f_best is very high and mu is even higher? No, mu is usually lower than f_best initially? No, f_best is the min observed. mu is GP mean. 
    # Initially, GP is uncertain, mu is roughly 0 (if standardized) or mean of data. 
    # If f_best is 10, and mu is 0, term = 0.5 * (10 - 0 - 0.5*sigma) > 0.
    # If f_best is 0, and mu is 0, term = 0.5 * max(0, -0.5*sigma) = 0.
    # So when we find the minimum, scores become 0. This is bad for continuing to search?
    # If f_best is the global min, we want to stop or explore less.
    # But we need to return finite scores.
    
    # If scores are all 0, we might want to fall back to exploration (sigma / cost).
    # Let's check if the max score is close to 0.
    if np.max(scores) < 1e-6:
        # Fall back to exploration: prefer high sigma, low cost
        scores = safe_sigma / safe_cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
