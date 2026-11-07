import numpy as np
from scipy.stats import norm
import time

_STEP_COUNT = 0
_TIMEOUT = 290.0  # Safety margin before the 330s kill
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    global _STEP_COUNT, _START_TIME

    # Deadline guard: if we are close to the time limit, return a fast exploit-only score
    if time.time() - _START_TIME > _TIMEOUT:
        return -mu

    _STEP_COUNT += 1

    # Ensure sigma is positive to avoid division by zero or negative sqrt issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Local Intensive Expected Improvement (LIEI)
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # LIEI clips the improvement term to be non-negative:
    # improvement = max(f_best - mu, 0)
    # This forces the score to be 0 (plus the small sigma term which is also suppressed
    # if we treat it as part of the improvement logic or just standard EI with clipped improvement)
    
    # Let's look at the specific definition suggested:
    # "clipping the improvement term (f_best - mu) to be non-negative before applying the standard EI formula"
    # Standard EI formula is: E[max(0, f_best - X)]
    # If we clip (f_best - mu) to be non-negative, say I = max(f_best - mu, 0),
    # does that mean we compute I * Phi(z) + sigma * phi(z)?
    # If f_best - mu < 0, then I = 0.
    # Then the term becomes 0 * Phi(z) + sigma * phi(z) = sigma * phi(z).
    # This is still positive. This doesn't fully suppress exploration.
    
    # A more aggressive "Local Intensive" approach often used in BO for multimodal functions
    # is to simply use the probability of improvement or a modified EI that goes to 0 when mu > f_best.
    # However, the prompt specifically says: "clipping the improvement term (f_best - mu) to be non-negative before applying the standard EI formula".
    
    # Let's interpret "standard EI formula" as the analytic expression:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If we replace (f_best - mu) with max(f_best - mu, 0):
    # EI_LIEI = max(f_best - mu, 0) * Phi(z) + sigma * phi(z)
    
    # Let's check the behavior:
    # If mu > f_best (predicted worse than best):
    # max(f_best - mu, 0) = 0
    # EI_LIEI = 0 * Phi(z) + sigma * phi(z) = sigma * phi(z)
    # Since z = (f_best - mu) / sigma is negative, phi(z) is small but positive.
    # This is still an exploration term.
    
    # Perhaps the intent is to suppress the exploration term as well?
    # "forcing the search to focus exclusively on high-probability improvement zones"
    # If we want to focus on improvement zones, we might want the score to drop off sharply
    # when mu > f_best.
    
    # Alternative interpretation:
    # Maybe the "improvement term" refers to the entire expectation?
    # No, "clipping the improvement term (f_best - mu)".
    
    # Let's try a different structural change that fits "Local Intensive":
    # Use a "Probabilistic Improvement" or a "Sharp" EI.
    # Actually, a common technique for multimodal functions is to use a smaller length scale
    # or a more aggressive exploitation.
    
    # Let's stick to the prompt's specific instruction first, but maybe modify the sigma term?
    # If we use:
    # improvement = np.maximum(f_best - mu, 0.0)
    # z = improvement / sigma_safe  # This changes z too?
    # No, z is usually defined with the raw difference.
    
    # Let's try:
    # diff = f_best - mu
    # z = diff / sigma_safe
    # # Clip the diff term
    # diff_clipped = np.maximum(diff, 0.0)
    # ei = diff_clipped * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # This still has the sigma * pdf(z) term.
    # To "force search to focus exclusively on high-probability improvement zones",
    # we might want to multiply the whole thing by a mask?
    # Or use a different acquisition like UCB with a very small beta?
    
    # Let's look at the score 0.0380. It's very low.
    # The parent was 0.53. The change to "Penalized EI" with lambda=0.05 dropped it significantly.
    # This suggests that penalizing sigma (uncertainty) was harmful.
    # The previous best (1.0000) likely used standard EI or a variant that didn't penalize sigma.
    
    # The directive says: "Make a STRUCTURAL change... a different design or modelling family...".
    # "Strategy to try in this change: Implement a Local Intensive Expected Improvement (LIEI) by clipping the improvement term (f_best - mu) to be non-negative..."
    
    # Let's implement exactly that, but also consider that the `sigma * pdf(z)` term
    # might be the culprit for exploring bad regions.
    # If we clip `diff`, the term `sigma * pdf(z)` remains.
    # However, if `diff` is negative, `z` is negative. `pdf(z)` is symmetric, so it's positive.
    # So we still get a positive score for regions where mu > f_best.
    
    # To truly suppress exploration in non-improvement zones, we could multiply the
    # exploration term by the CDF of the improvement?
    # Or simply:
    # score = np.maximum(f_best - mu, 0) * norm.cdf(z) + sigma * norm.pdf(z) * norm.cdf(z)
    # This would make the exploration term vanish when z is very negative.
    
    # But let's follow the prompt's literal instruction first, as it's a specific strategy.
    # If it fails, we can iterate. But I need to provide the best shot now.
    
    # Actually, looking at the "Inspiration 3" (score 0.9090), it mentions "smooth surrogates".
    # The current task is a fixed seeded random RBF mixture.
    
    # Let's try a "Sharp" EI or "LIEI" as described.
    # diff = f_best - mu
    # z = diff / sigma_safe
    # diff_pos = np.maximum(diff, 0.0)
    # score = diff_pos * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Let's also consider that the global _STEP_COUNT might be useful for adaptation.
    # If we are in the early steps, we might want more exploration.
    # But the directive says "do not micro-tune constants".
    
    # Let's try a slightly different structural change:
    # Instead of EI, use **Probability of Improvement (PI)**?
    # PI = Phi(z). This is bounded [0,1].
    # For minimization, if mu < f_best, z > 0, PI > 0.5.
    # If mu > f_best, z < 0, PI < 0.5.
    # This is a very exploitative acquisition. It ignores the magnitude of improvement.
    # For a multimodal function with a known minimum, this might be too greedy and get stuck.
    
    # What about **Upper Confidence Bound (UCB)**?
    # UCB = -mu + kappa * sigma
    # For minimization.
    # This is a different family.
    
    # The directive specifically suggests LIEI. Let's trust the directive.
    # But let's refine the LIEI interpretation to be more effective.
    # If we use:
    # score = np.maximum(f_best - mu, 0) * norm.cdf(z) + sigma * norm.pdf(z)
    # The second term is still there.
    
    # Let's try to zero out the exploration term when mu > f_best.
    # mask = (f_best - mu) > 0
    # score = np.where(mask, (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
    # This forces the score to be exactly 0 for any point predicted to be worse than best.
    # This is a very strong "Local Intensive" strategy. It only considers points
    # where the GP mean is already better than the best found.
    # This might be too restrictive early on, but the problem says "fixed seeded random RBF mixture",
    # so the landscape is static.
    
    # Let's go with this masked EI. It is a structural change from standard EI.
    
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Standard EI terms
    ei_term = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Mask: only keep positive improvement potential
    # If diff <= 0, score is 0.
    # If diff > 0, score is standard EI.
    score = np.where(diff > 0, ei_term, 0.0)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
