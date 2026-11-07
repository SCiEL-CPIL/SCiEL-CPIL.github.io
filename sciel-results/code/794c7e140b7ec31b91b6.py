"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0


def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1

    # Deadline guard: if we are too close to the limit, return a simple score
    # to ensure we don't timeout.
    if time.time() - _start_time > 325:
        # Fallback to a simple greedy score
        return -mu

    # t is the number of optimization steps taken (excluding initial 5)
    # _call_count is 1 for the first optimization step
    t = _call_count

    # Handle case where sigma is very small or zero to avoid division by zero
    # For numerical stability, add a small epsilon to sigma.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize EI.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Knowledge Gradient (KG) approximation
    # KG approximates the expected reduction in the minimum value.
    # A common approximation for KG in the context of a single evaluation is
    # related to the expected improvement but with a different weighting.
    # However, exact KG is expensive.
    #
    # Let's use a known approximation or a variant.
    # One simple form of the "Expected Improvement" is actually a lower bound on KG.
    # Another approach is to use the "Probability of Improvement" (PI) blended with EI.
    #
    # The directive suggests replacing EI with KG.
    # A practical approximation for KG (often called "KG-1" or similar in literature)
    # for a single candidate x is:
    # KG(x) = E[max(0, f_best - f(x))]  which is exactly EI.
    #
    # However, a more refined KG considers the probability that x is the new minimum
    # and the expected improvement given that.
    #
    # Let's try a blend of EI and a "confidence" term.
    # Or, as suggested in the cell findings: "Hybrid EI with adaptive exploration bonus".
    # The finding states: "square-root exploration term provides smoother exploration than linear sigma"
    # and improved score from 0.02675 -> 0.06009.
    #
    # Let's incorporate an adaptive exploration bonus based on sigma.
    # score = EI + lambda * sigma^alpha
    # where lambda and alpha are tuned.
    # The finding mentions square-root, so alpha = 0.5.
    #
    # Let's try: score = EI + 0.1 * sqrt(sigma_safe)
    # Or perhaps a dynamic lambda based on t?
    # Early on, we might want more exploration.
    #
    # Let's look at the "inspiration 3" which tried a hybrid but ended up just returning EI.
    # The current program is basically EI.
    #
    # Let's try the KG approximation more directly.
    # There is a closed-form approximation for KG when the posterior is Gaussian.
    # KG(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)  ... this is EI.
    #
    # Actually, the exact KG for a single step is:
    # KG(x) = E[ min( f_best, f(x) ) ] - min( f_best, E[f(x)] ) ... no.
    # KG(x) = E[ max(0, f_best - f(x)) ] is EI.
    #
    # Let's stick to the "Hybrid EI with adaptive exploration bonus" idea from the findings.
    # The finding says: "Hybrid EI with adaptive exploration bonus better balances exploitation and exploration... square-root exploration term provides smoother exploration".
    #
    # Let's implement: score = EI + beta * sqrt(sigma)
    # We need to choose beta.
    # If sigma is in objective units, sqrt(sigma) might be small.
    # Let's try beta = 0.1.
    #
    # Also, we can normalize the scores to ensure they are comparable?
    # The contract says "higher meaning evaluate next".
    #
    # Let's try:
    # exploration_bonus = 0.1 * np.sqrt(sigma_safe)
    # score = ei + exploration_bonus
    #
    # However, EI can be negative if mu > f_best (which shouldn't happen much if f_best is min, but mu can be higher).
    # Actually, EI is always >= 0 because if mu > f_best, z is negative, Phi(z) is small, but (f_best-mu) is negative.
    # Wait, (f_best - mu) is negative. Phi(z) is positive. So the first term is negative.
    # The second term sigma * phi(z) is positive.
    # It can be shown that EI >= 0.
    #
    # Let's add the exploration bonus.
    # To make it adaptive, maybe beta decreases with t?
    # beta = 0.1 / (1 + t)
    #
    # Let's try a fixed beta first.
    #
    # Another idea from the directive: "soft-min/soft-max via log-sum-exp".
    # This is usually for multi-objective or combining multiple objectives.
    # Here we have a single objective.
    #
    # Let's try the KG approximation using the "Expected Improvement" but with a "temperature" parameter.
    # Or simply the hybrid.
    #
    # Let's go with the hybrid EI + sqrt(sigma) bonus.
    
    beta = 0.1
    exploration_bonus = beta * np.sqrt(sigma_safe)
    
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
