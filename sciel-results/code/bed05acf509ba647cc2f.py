import numpy as np
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s

# Known global minimum of the base function
GLOBAL_MIN = -0.8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    
    Strategy:
    1. Sample from the posterior: y = mu - sigma * Z (since we minimize, lower y is better).
       Z ~ N(0, 1).
    2. The "value" of a point is the expected improvement over the global minimum,
       penalized by cost.
    3. We want to maximize the score. Since we want low function values, we can
       define the score as (GLOBAL_MIN - y_sample) / cost.
       If y_sample < GLOBAL_MIN, score is positive (great).
       If y_sample > GLOBAL_MIN, score is negative.
       
       Alternatively, a more robust approach for TS in minimization is to sample
       the function value and then rank them. However, we need a scalar score.
       
       Let's use: Score = (f_best - y_sample) / cost.
       This measures the improvement over the current best, adjusted for cost.
       But f_best might be far from the global min.
       
       Better: Use the expected improvement over the global minimum, but since we
       are sampling, we can just use the sample itself.
       
       Let's try: Score = (GLOBAL_MIN - y_sample) / cost.
       This encourages points where the sample is lower than -0.8.
       
       Wait, the objective is to minimize. The evaluator returns the raw value.
       The acquisition function returns scores where higher = evaluate next.
       
       If we sample y ~ N(mu, sigma^2), then a low y is good.
       So we want to maximize -y / cost? Or (GLOBAL_MIN - y) / cost?
       
       If y is very low (good), (GLOBAL_MIN - y) is large positive.
       If y is high (bad), (GLOBAL_MIN - y) is negative.
       
       This seems reasonable. It directly targets the global minimum.
       
       However, early on, f_best might be much worse than GLOBAL_MIN.
       Using GLOBAL_MIN as a reference might be too aggressive if the GP is uncertain.
       
       Alternative: Score = (f_best - y_sample) / cost.
       This is the improvement over the current best.
       
       Let's combine: Use a reference point that is the minimum of f_best and GLOBAL_MIN?
       No, f_best is the observed min. GLOBAL_MIN is the theoretical min.
       
       Let's stick to the standard Thompson Sampling for minimization:
       Sample y_i = mu_i - sigma_i * Z_i.
       Select the point with the minimum y_i.
       But we must return scores for all 2000 points.
       
       We can define the score as the probability that this point is the best, or
       simply the negative of the sampled value, adjusted for cost.
       
       Score_i = -y_i / cost_i
       
       This is simple and effective. It favors low sampled values and low costs.
       
       Let's add a small exploration bonus if needed, but TS handles it via sigma.
       
       To ensure finite scores and avoid division by zero:
       cost_safe = cost + 1e-9
       sigma_safe = sigma + 1e-9
       
       Z is a standard normal sample. We can use a fixed Z or random?
       Using random Z in an acquisition function is standard for TS.
       However, the function is called repeatedly. Randomness is fine.
       
       One issue: If we use random Z, the scores change every call. That's expected for TS.
       
       Let's implement:
       Z = np.random.randn(2000)
       y = mu - sigma_safe * Z
       score = -y / cost_safe
       
       We should also handle the case where we want to exploit the global minimum knowledge.
       If we know the min is -0.8, we can shift the score to be relative to that.
       score = (GLOBAL_MIN - y) / cost_safe
       
       If y < GLOBAL_MIN, score > 0.
       If y > GLOBAL_MIN, score < 0.
       
       This might be better because it doesn't depend on f_best which might be stuck in a local minimum.
       
       Let's try this.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Deterministic UCL
        sigma_safe = sigma + 1e-9
        cost_safe = cost + 1e-9
        ucl = -(mu - 1.0 * sigma_safe)
        return ucl / cost_safe

    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9
    
    # Thompson Sampling: Sample from posterior
    # We want to minimize, so lower samples are better.
    # Z ~ N(0, 1)
    Z = np.random.randn(len(mu))
    
    # Sampled function values
    y_samples = mu - sigma_safe * Z
    
    # Score: How much better is this sample than the global minimum?
    # Higher score = better.
    # If y_sample < GLOBAL_MIN, score is positive.
    # If y_sample > GLOBAL_MIN, score is negative.
    # Dividing by cost makes it cost-aware.
    scores = (GLOBAL_MIN - y_samples) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
