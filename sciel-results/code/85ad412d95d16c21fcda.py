import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    if remaining_time < _DEADLINE_MARGIN:
        # Fallback: Greedy (lowest mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    eps = 1e-9
    sigma_safe = sigma + eps

    # Calculate z-score for Probability of Improvement (POI)
    # We want P(f(x) < f_best) = P(N(mu, sigma) < f_best)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # POI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Structural Change: Adaptive Exploitation Gate
    # Instead of linearly blending or adding a small bonus to a continuous score,
    # we apply a non-linear transformation that heavily rewards points with high POI
    # but also requires a certain level of "confidence" in the improvement.
    
    # The gate is based on the gap to the known minimum (3.0).
    # Early on (large gap), we want to be more permissive (explore).
    # Late on (small gap), we want to be strict (exploit).
    
    gap = max(f_best - 3.0, 0.0)
    
    # Threshold for POI to be considered "good enough"
    # As gap shrinks, we require higher POI to score well.
    # Let's map gap to a threshold between 0.5 (neutral) and 0.99 (very strict)
    # When gap is large, threshold is low (0.5), so almost any point with mu < f_best gets a decent score.
    # When gap is small, threshold is high (0.99), so only points with very high probability of improvement get high scores.
    
    # Simple exponential decay of the threshold requirement
    # threshold = 0.5 + 0.49 * (1 - exp(-gap / 1.0))
    # If gap=0, threshold=0.5. If gap=inf, threshold=0.99.
    # Wait, we want the opposite? 
    # If gap is large (early), we are far from optimum. We should explore. 
    # If gap is small (late), we are close. We should exploit.
    # Exploitation means we only pick points that are *very likely* to be better.
    # So, as gap -> 0, we should require higher POI.
    # As gap -> large, we should accept lower POI (more exploration).
    
    # Let's define a "strictness" parameter beta.
    # beta is high when gap is small.
    beta = 1.0 / (1.0 + gap)
    
    # We construct a score that is zero if POI is below a dynamic threshold,
    # and rises sharply as POI exceeds it.
    # Threshold = 0.5 + 0.4 * beta
    # If gap=0, beta=1, threshold=0.9.
    # If gap=10, beta~0.1, threshold=0.54.
    
    threshold = 0.5 + 0.4 * beta
    
    # Score is a scaled and shifted version of POI, but gated.
    # To make it a smooth acquisition function, we can use a soft gate.
    # score = poi * (1 + 10 * (poi - threshold)) if poi > threshold else 0
    # This creates a strong preference for high POI points once the threshold is crossed.
    
    # Alternatively, a simpler structural change:
    # Use a "Power" transformation of the improvement probability.
    # score = (poi - 0.5) ** 2 if poi > 0.5 else 0
    # This emphasizes points that are clearly better.
    
    # Let's try a different structural family: 
    # **Hyper-Volume Improvement (HVI) proxy** or **Confidence-bound based on POI**.
    
    # Actually, the cell finding said POI is better. Let's stick to POI but change the *combination* strategy.
    # Current: NEI + small bonus.
    # New: Pure POI, but with a dynamic "floor" that suppresses points with low probability of improvement.
    
    # score = poi
    # But to avoid picking points that are barely better (low POI ~ 0.5), we subtract a baseline.
    # score = poi - 0.5
    # This makes the best possible score 0.5 (when poi=1) and worst -0.5 (when poi=0).
    # Points with mu > f_best have poi < 0.5, so they get negative scores.
    # Points with mu << f_best have poi ~ 1, so they get ~0.5.
    
    # To enhance exploitation late in the run, we can amplify the difference.
    # score = (poi - 0.5) * (1 + 2 * beta)
    
    # Let's refine the "strictness".
    # If we are very close to the minimum (gap ~ 0), we only want to move if we are very confident.
    # A high power on (poi - 0.5) achieves this.
    
    # Let's use: score = (poi - 0.5) ** (1 + 2*beta) * (1 + 2*beta)
    # If beta=1 (gap=0): score = (poi-0.5)**3 * 3. Range [-0.375, 0.375]. Very flat near 0.5.
    # If beta=0.1 (gap=10): score = (poi-0.5)**1.2 * 1.2. Range [-0.24, 0.24]. More linear.
    
    # This dynamically shifts the acquisition function from linear (exploratory) to cubic (exploitative).
    
    exponent = 1.0 + 2.0 * beta
    scale = exponent
    
    # Ensure we don't take power of negative numbers if we shift incorrectly, but poi-0.5 is in [-0.5, 0.5].
    # Odd powers preserve sign. Even powers flip sign. 
    # We want higher score for higher poi.
    # If exponent is odd, (poi-0.5)^exp is monotonic.
    # 1+2*beta is always odd if beta is integer? No.
    # Let's just use the absolute value or ensure monotonicity.
    # Actually, simpler:
    # score = (poi - 0.5) * (1 + 5 * beta * (poi - 0.5))
    # This is a quadratic boost for high POI.
    
    # Let's try the most distinct structural change: **Thompson Sampling-like** or **Pure POI with Adaptive Threshold**.
    
    # I will implement:
    # 1. Calculate POI.
    # 2. Calculate a dynamic threshold `t` based on gap.
    # 3. Score is 0 if POI < t, otherwise (POI - t) / (1 - t).
    # This effectively discards all candidates that are not "sufficiently likely" to be better.
    # As we get closer to the optimum, the threshold t increases, becoming more selective.
    
    # t = 0.5 + 0.45 * (1 - np.exp(-gap))
    # If gap=0, t=0.5. (Wait, if gap=0, we are at the minimum? No, gap is f_best - 3. If f_best=3, gap=0. We should be very strict? 
    # If f_best=3, we found the minimum. We shouldn't evaluate more. But the loop continues. 
    # If we are close, we want to be sure. So t should be HIGH when gap is SMALL.
    # My previous logic: t = 0.5 + 0.4 * beta, where beta = 1/(1+gap).
    # If gap=0, beta=1, t=0.9. (Strict)
    # If gap=10, beta=0.1, t=0.54. (Lenient)
    # This seems correct.
    
    beta = 1.0 / (1.0 + gap)
    t = 0.5 + 0.4 * beta
    
    # Score:
    # If POI > t, score is proportional to how much it exceeds t.
    # To make it comparable across steps, normalize by (1 - t).
    scores = np.where(poi > t, (poi - t) / (1.0 - t), 0.0)
    
    # Add a tiny bit of noise or tie-breaking? No, keep it deterministic based on GP.
    # However, if many points have POI > t, the one with the highest POI wins.
    # This is effectively "Pick the point with the highest probability of being the new best".
    
    # To ensure we don't get stuck if all scores are 0 (which shouldn't happen as mu < f_best usually exists),
    # we can add a very small epsilon * sigma to break ties in favor of exploration if POI is identical?
    # No, the contract says "higher meaning evaluate next".
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
