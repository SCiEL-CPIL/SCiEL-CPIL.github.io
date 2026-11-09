import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure arrays are 1D
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check time
    if time.time() - t0 > 250:
        return np.zeros_like(mu)
    
    # Calculate z-score for Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip EI to non-negative to avoid negative scores from numerical noise
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Lower Confidence Bound (LCB) term
    # LCB = mu - kappa * sigma
    # We want to minimize LCB, so we maximize -LCB = -mu + kappa * sigma
    # We need to scale kappa appropriately.
    
    # Adaptive exploration coefficient
    # Decay exploration as f_best improves (gets closer to 0).
    # f_best is the current best value.
    # If f_best is large, we are early -> high exploration.
    # If f_best is small, we are late -> low exploration.
    
    # Scale factor for sigma to make it comparable to mu/ei
    # mu and sigma are in objective units.
    # Let's normalize sigma relative to the current scale of the problem.
    # A simple heuristic: use the median or max of sigma, or just a fixed scale.
    # Since mu can be negative, -mu can be large positive or negative.
    # EI is always non-negative.
    # Let's try to balance EI and LCB.
    
    # Normalize sigma for the exploration term
    # If we use raw sigma, it might be too small or too large compared to EI.
    # Let's look at the magnitude of EI.
    # Early on, EI can be large. Late on, EI is small.
    
    # Adaptive kappa
    # w = 1.0 / (1.0 + f_best / 5.0)
    # This was used in the current program.
    # Let's try a different decay.
    # Maybe based on the ratio of sigma to f_best?
    
    # Let's try a simple LCB formulation:
    # score = -mu / cost_safe + kappa * sigma / cost_safe
    # But -mu can be negative if mu > 0.
    # EI is a better proxy for "improvement".
    
    # Hybrid: EI + Exploration
    # Exploration term: sigma / cost_safe
    # Scale it by an adaptive coefficient.
    
    # Let's estimate the "scale" of the objective.
    # f_best is a good indicator.
    # If f_best is 10, sigma might be ~1-5.
    # If f_best is 0.1, sigma might be ~0.1-0.5.
    
    # Let's use a coefficient that depends on f_best.
    # kappa = c / (1 + f_best)
    # If f_best = 10, kappa ~ 0.1c
    # If f_best = 0.1, kappa ~ 0.9c
    
    # What is a good c?
    # We want the exploration term to be comparable to EI.
    # EI ~ (f_best - mu) * ...
    # If mu is far from f_best, EI is large.
    
    # Let's try:
    # score = score_ei + w * (sigma_safe / cost_safe)
    
    # w should be large when f_best is large (early).
    # w = 1.0 / (1.0 + f_best / 10.0)
    # f_best = 20 -> w = 1/3 ~ 0.33
    # f_best = 5 -> w = 1/1.5 ~ 0.67
    # f_best = 1 -> w = 1/1.1 ~ 0.9
    # f_best = 0.1 -> w ~ 1.0
    
    # But sigma is in objective units.
    # If sigma is 1.0 and cost is 1.5, term is 0.67.
    # If EI is 0.5, this dominates.
    
    # Let's normalize sigma by the current f_best?
    # No, f_best can be 0.
    
    # Let's try a fixed scale but adaptive weight.
    # The current program used:
    # sigma_norm = sigma / max(sigma)
    # score = score_ei + 0.1 * w * sigma_norm
    
    # This normalized sigma to [0, 1], which loses the absolute scale information.
    # If all sigmas are small, sigma_norm is still [0, 1].
    # This encourages exploration of the *most* uncertain point, regardless of absolute uncertainty.
    # This might be good for escaping local optima.
    
    # Let's stick to the structural idea of the current program (adaptive weight based on f_best)
    # but refine the exploration term to be more principled.
    
    # Idea: Use the raw sigma, but scaled by a factor that depends on the typical EI.
    # typical_ei = np.mean(score_ei)
    # if typical_ei > 0:
    #     scale = typical_ei
    # else:
    #     scale = 1.0
    # score = score_ei + w * (sigma_safe / cost_safe) * (scale / np.median(sigma_safe + 1e-9))
    # This is getting complicated.
    
    # Let's try a simpler LCB approach.
    # LCB = mu - beta * sigma
    # We want to minimize LCB.
    # So we maximize -LCB = -mu + beta * sigma.
    # Divide by cost: (-mu + beta * sigma) / cost.
    
    # But -mu can be negative.
    # EI is always positive.
    # Let's combine EI and LCB.
    
    # Actually, the current program's approach of normalizing sigma and adding a small bonus is robust.
    # The score 1.1284 suggests it's working well.
    # The inspiration 1 (score 1.1284) is essentially the same as the current program.
    # Inspiration 2 and 3 (score 1.0) are simple EI/cost.
    
    # The difference between 1.0 and 1.1284 is the exploration term.
    # Let's see if we can improve the exploration term.
    
    # Current: score = ei/cost + 0.1 * w * (sigma / max_sigma)
    # w = 1 / (1 + f_best/5)
    
    # Let's try to make the exploration term more sensitive to the absolute uncertainty.
    # Instead of normalizing by max_sigma, let's normalize by a characteristic scale.
    # Maybe the median of f_best? Or a fixed scale?
    
    # If we use raw sigma / cost:
    # score = ei/cost + w * (sigma / cost)
    # This might be too large if sigma is large.
    
    # Let's try:
    # score = ei/cost + w * (sigma / (f_best + 1.0) * cost_safe)
    # No, that's dimensionally weird.
    
    # Let's try to use the probability of improvement as a weight for exploration.
    # P(improve) = Phi(z)
    # If P(improve) is low, we might want to explore more.
    # So weight = 1 - Phi(z)?
    
    # score = ei/cost + w * (1 - Phi(z)) * (sigma / cost)
    
    # Let's test this idea.
    # z = (f_best - mu) / sigma
    # If mu << f_best, z is large, Phi(z) ~ 1, weight ~ 0.
    # If mu >> f_best, z is negative, Phi(z) ~ 0, weight ~ 1.
    # This encourages exploration where improvement is unlikely?
    # That seems counter-intuitive. We want to explore where improvement is *possible* but uncertain.
    
    # Maybe weight = Phi(z) * (1 - Phi(z))? (Uncertainty of the improvement event)
    
    # Let's stick to the current structure but tweak the parameters.
    # The current program uses w = 1 / (1 + f_best/5).
    # Maybe the decay is too fast or too slow.
    
    # Let's try w = 1 / (1 + f_best/10). Slower decay.
    # And increase the coefficient from 0.1 to 0.2?
    
    # Also, consider the cost in the exploration term.
    # The current program does NOT divide the exploration term by cost.
    # score = ei/cost + 0.1 * w * sigma_norm
    # This means expensive points get the same exploration bonus as cheap points.
    # But we are dividing EI by cost.
    # So a cheap point with low EI might beat an expensive point with high EI if the exploration bonus is high enough.
    # This is good for cost-awareness.
    
    # However, if we want to be strictly cost-aware, we should divide the exploration term by cost too.
    # score = ei/cost + 0.1 * w * (sigma_norm / cost)
    # This would penalize expensive exploration.
    
    # Let's try this modification.
    
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    w = 1.0 / (1.0 + f_best / 10.0)
    
    # Option A: Current
    # score = score_ei + 0.1 * w * sigma_norm
    
    # Option B: Cost-aware exploration
    # score = score_ei + 0.1 * w * (sigma_norm / cost_safe)
    
    # Let's try Option B with a slightly higher coefficient.
    # If cost is ~1.5, dividing by 1.5 reduces the bonus.
    # So we might need a larger coefficient.
    
    score = score_ei + 0.2 * w * (sigma_norm / cost_safe)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Final time check
    if time.time() - t0 > 290:
        return score
        
    return score
