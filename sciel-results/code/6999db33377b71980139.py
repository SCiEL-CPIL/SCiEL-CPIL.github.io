"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) style acquisition function.
    
    KG estimates the expected improvement in the global minimum.
    Approximation:
    1. Compute EI for each candidate (benefit if it's the new best).
    2. Estimate probability that candidate is the new minimum:
       p_new_min = Phi((f_best - mu) / sigma)
       This is the probability that f(x) < f_best.
       However, for KG we want the probability that this point is the *global* minimum.
       A common approximation for the value of information is related to the probability
       of being the best *and* the magnitude of improvement.
       
       Another view: KG approximates E[min(f_best, f(x)) - min(f(x))]
       Actually, KG is E[min(f_best, f(x)) - min_{all} f].
       A simpler surrogate that captures "value of information" is:
       Score = E[max(0, f_best - f(x))] * P(f(x) is likely to be significant)
       
       However, a direct implementation of KG is expensive. 
       A common approximation for KG in BO is to use the Expected Improvement (EI)
       but scaled by a factor that reflects the "surprise" or the potential to be the new global best.
       
       The directive suggests: "approximating the probability that each candidate is the new minimum, scaled by a sublinear cost discount (1/sqrt(cost))"
       
       Let's implement:
       1. EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       2. P_new_min = Phi((f_best - mu) / sigma) -- This is just PoI.
          But PoI is already part of EI structure.
          
          Let's look at the "Knowledge Gradient" more carefully.
          KG(x) = E[ max(0, f_best - min(f(x), min_{other} f)) ]
          This is hard to compute.
          
          A simpler proxy for "is this a new global minimum" is how much it beats f_best.
          If mu is much lower than f_best, it's likely a new minimum.
          
          Let's try the specific suggestion:
          Score = EI * (1 / sqrt(cost))
          Wait, the current best is (EI + 0.1*UCB)/cost.
          The suggestion says "scaled by a sublinear cost discount (1/sqrt(cost))".
          This implies the cost penalty is less severe than 1/cost.
          This would favor high-cost points more than 1/cost does.
          
          Let's try: Score = EI / sqrt(cost)
          Or Score = (EI + w*UCB) / sqrt(cost)
          
          Let's stick to the directive: "approximating the probability that each candidate is the new minimum"
          Let p = Phi((f_best - mu)/sigma)
          Let benefit = (f_best - mu) * p + sigma * pdf(z) (This is EI)
          
          Maybe the "probability that each candidate is the new minimum" is meant to be a gate?
          Or maybe it's the KG value itself?
          
          Let's try a simple KG approximation:
          KG ≈ EI * (1 + sigma) ? No.
          
          Let's try the sublinear cost discount as the primary structural change.
          Score = EI / np.sqrt(cost)
          
          Also, the directive mentions "approximating the probability that each candidate is the new minimum".
          Let's combine:
          Score = EI * p_new_min / np.sqrt(cost) ?
          Since EI is already ~ (f_best-mu)*p + ..., multiplying by p again might dampen it too much.
          
          Let's try: Score = (EI + 0.5 * UCB) / np.sqrt(cost)
          This changes the cost sensitivity from linear to sublinear.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. UCB
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # 2. EI
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # 3. Cost discount: sublinear (1/sqrt(cost)) instead of linear (1/cost)
    # This encourages exploring more expensive regions if they offer high EI/UCB
    cost_safe = np.maximum(cost, 1e-9)
    discount = 1.0 / np.sqrt(cost_safe)
    
    # Combine EI and UCB with sublinear cost discount
    score = (ei + 0.5 * ucb) * discount
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
