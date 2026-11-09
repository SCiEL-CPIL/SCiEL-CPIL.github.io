"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
        _call_count = 0
    
    _call_count += 1
    elapsed = time.time() - _start_time
    
    # Safety margin: return simple EI if we are dangerously close to the 330s limit
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-8)

    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Knowledge Gradient (KG) approximation:
    # Probability that this point is the new minimum
    # This is P(mu < f_best) = Phi(z)
    prob_new_min = norm.cdf(z)
    
    # Expected reduction in f_best if it is the new minimum
    # E[f | f < f_best] = mu - sigma * L(z) where L is inverse Mills ratio
    # Actually, E[min(f, f_best)] = mu * Phi(z) + sigma * phi(z) + f_best * Phi(-z)
    # So E[f_best - min(f, f_best)] = f_best - mu*Phi(z) - sigma*phi(z) - f_best*Phi(-z)
    # = f_best * Phi(z) - mu*Phi(z) - sigma*phi(z)
    # = (f_best - mu) * Phi(z) - sigma * phi(z)
    # Wait, EI = E[max(f_best - f, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Let's re-verify.
    # f ~ N(mu, sigma^2).
    # E[max(f_best - f, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma.
    # This is correct.
    
    # KG is essentially EI but sometimes with different weighting or interpretation.
    # A common approximation for KG in BO is to use EI directly, or a modified version.
    # The directive suggests: "estimating the expected improvement in the best observed value as the sum of the probability that the current candidate is the new global minimum (via the GP predictive distribution) and the expected reduction in f_best if it is".
    # This decomposition is exactly EI.
    # However, to make it a "structural change" as requested (different design/modelling family), 
    # and given that the current program uses UCB (-mu + beta*sigma/cost), switching to a cost-aware EI/KG hybrid is a significant change.
    
    # Let's try a hybrid: Cost-aware EI.
    # score = EI / cost
    # This directly optimizes improvement per unit cost.
    
    # To add the "KG" flavor, we can weight the exploration part differently.
    # But let's stick to a robust Cost-aware EI first, as it's a standard and strong baseline.
    # The previous program was UCB. This is EI.
    
    # We can also blend UCB and EI.
    # But let's try pure Cost-aware EI with a dynamic beta if we want to keep exploration.
    # Actually, EI already balances exploration and exploitation via the z-score.
    
    # Let's use Cost-aware EI.
    score = ei / np.maximum(cost, 1e-8)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
