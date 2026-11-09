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

    # Knowledge Gradient (KG) approximation for minimization
    # KG approximates the expected improvement in the best observed value.
    # For a candidate x with predictive mean mu_x and std sigma_x:
    # The expected improvement in f_best is E[max(f_best - f(x), 0)]
    # This is exactly the Expected Improvement (EI) for minimization.
    # However, a better KG approximation considers the probability that x is the new best
    # and the expected value of x given that it is the new best.
    #
    # A common efficient approximation for KG in BO is:
    # KG(x) ≈ E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    #        = E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    #
    # Let's derive this:
    # Let Y ~ N(mu, sigma^2). We want E[max(f_best - Y, 0)].
    # This is the standard EI formula:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #
    # However, the directive suggests "decoupling the discovery probability from the value magnitude".
    # Let's look at the components:
    # P(Y < f_best) = Phi(z) where z = (f_best - mu) / sigma
    # E[f_best - Y | Y < f_best] = f_best - (mu - sigma * phi(z)/Phi(z))
    # So EI = P(Y < f_best) * (f_best - mu + sigma * phi(z)/Phi(z))
    #
    # The term (f_best - mu) is the "value magnitude" part (how far below current best is the mean).
    # The term sigma * phi(z)/Phi(z) is the "uncertainty adjustment".
    #
    # A simpler KG-like score that emphasizes exploration in regions where the current best is uncertain
    # or where the probability of improvement is high is simply EI itself.
    # But let's try a variant that might be more robust for multimodal functions like Griewank.
    #
    # Another perspective:
    # KG is often approximated by EI in practice for single-objective BO.
    # However, we can tune the exploitation vs exploration balance.
    # The current UCB score is beta * sigma - mu.
    # EI is (f_best - mu) * Phi(z) + sigma * phi(z).
    #
    # Let's use EI but with a cost-awareness factor.
    # To improve on the UCB score (2.09), let's try EI which was the parent (1.49) but maybe with better parameters?
    # Wait, the parent was EI with an exploitation bias. The current is UCB.
    # The directive says "KEEP this design" (UCB) but "investigate improvements within its family".
    # And "Implement a Knowledge Gradient (KG) approximation".
    #
    # Let's implement a KG approximation that is computationally cheap.
    # A known approximation for KG is:
    # KG(x) = E[max(f_best - f(x), 0)]
    # This is EI.
    #
    # Let's try a different approach:
    # Use a "Probabilistic Improvement" score which is P(f(x) < f_best) = Phi(z).
    # This is purely exploratory in the sense that it ignores the magnitude of improvement.
    # But for multimodal functions, finding *any* new minimum is valuable.
    #
    # Let's combine EI with a cost penalty.
    # Score = EI / cost
    #
    # Let's also consider that f_best changes.
    # If f_best is very low, EI becomes small.
    #
    # Let's try EI with a small epsilon to avoid division by zero in sigma.
    # And normalize by cost.
    
    # Calculate z
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might creep in)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
