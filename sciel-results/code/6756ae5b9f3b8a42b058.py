import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Knowledge Gradient (KG) approximation:
    # The core idea is to weight the improvement by the probability that this
    # point is actually the new minimum, rather than just the expected magnitude.
    #
    # 1. Discovery Probability:
    #    P(x is new min) is approximated by the probability that f(x) < f_best.
    #    This is norm.cdf(z).
    #    We use this to gate the exploitation bias, ensuring we only strongly
    #    exploit points that are likely to be improvements.
    
    prob_improve = norm.cdf(z)
    
    # 2. Expected Value of Information (EVI) component:
    #    Standard EI is (f_best - mu) * cdf(z) + sigma * pdf(z).
    #    The term (f_best - mu) * cdf(z) is the expected improvement magnitude.
    #    The term sigma * pdf(z) is the uncertainty bonus.
    #    
    #    In the KG-inspired view, we want to emphasize the "discovery" aspect.
    #    We construct a score that balances:
    #    a) The likelihood of finding a new minimum (prob_improve)
    #    b) The expected gain if we do find one
    #    
    #    A robust approximation for KG in this context is to use the EI itself
    #    but modulate it or combine it with a strong exploitation signal that
    #    is gated by the probability of improvement.
    #
    #    The current program uses: (ei + exploitation_bias) / cost_damped
    #    The inspiration uses: (ei + exploitation_bias) / cost
    #
    #    Let's refine the exploitation bias. Instead of a simple linear -0.1 * mu,
    #    we can use a bias that is stronger when we are confident (low sigma) and 
    #    the point is likely an improvement (high prob_improve).
    #    
    #    However, the "linear UCB" finding suggests that a simple linear combination
    #    of mu and sigma works well. Let's stick to a linear form but tune the
    #    coefficients to reflect the KG intuition: prioritize low mu (exploitation)
    #    and high sigma (exploration) but penalize cost more heavily if the 
    #    expected improvement is low.
    
    # Let's try a hybrid:
    # Score = (EI + alpha * (-mu) + beta * sigma) / cost_adjusted
    
    # The current best (1.5255) uses:
    # score = (ei - 0.1*mu) / (1.0 + 0.5*(cost-1.0))
    
    # The inspiration (1.4950) uses:
    # score = (ei - 0.2*mu) / cost
    
    # The KG strategy suggests decoupling discovery probability from value magnitude.
    # We can implement this by scaling the exploitation term by the probability of improvement.
    # If prob_improve is high, we exploit more. If low, we explore more (via EI's sigma term).
    
    exploitation_bias = -0.2 * mu * prob_improve
    
    # We also add a small exploration bonus that is independent of EI's implicit sigma term,
    # scaled by cost, to ensure we don't ignore expensive but high-uncertainty regions
    # if the EI is low due to high mu.
    # Actually, EI already has sigma * pdf(z). 
    # Let's keep it simple and robust.
    
    # Cost adjustment: 
    # The current program dampens cost: 1.0 + 0.5 * (cost - 1.0)
    # This effectively reduces the penalty for cost, making the search less sensitive to cost.
    # Since the minimum is in the inner 80% (higher cost), damping cost helps find the minimum.
    # Let's keep this dampening but perhaps make it slightly stronger to further encourage
    # exploration in the central region.
    
    cost_damped = 1.0 + 0.4 * (cost - 1.0)
    
    # Combine
    score = (ei + exploitation_bias) / cost_damped
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
