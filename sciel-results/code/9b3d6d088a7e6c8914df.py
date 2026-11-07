# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import erf

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill

def _norm_cdf_vec(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def _norm_pdf_vec(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def _ei_dynamic(mu, sigma, fb, alpha):
    """
    Expected Improvement with a dynamic threshold.
    Instead of using fb directly, we use a threshold that shifts
    based on the mean uncertainty.
    """
    s = np.maximum(sigma, 1e-8)
    # Dynamic threshold: lower the bar when uncertainty is high
    # alpha controls the strength of this exploration bias
    mean_sigma = np.mean(s)
    # We want to explore more if mean_sigma is large relative to fb?
    # Actually, for minimization, if fb is large, we are far from optimum.
    # Let's use a threshold that is slightly worse than fb when uncertainty is high,
    # or slightly better when uncertainty is low.
    # Standard EI: E[max(0, fb - Y)]
    # Dynamic EI: E[max(0, (fb - delta) - Y)] where delta depends on uncertainty.
    # If uncertainty is high, we might accept a point that is not strictly better than fb
    # but has high potential. This is tricky.
    
    # Alternative interpretation from directive:
    # "Model the acquisition as a probabilistic 'zero-crossing' likelihood...
    # scoring candidates using the probability that the posterior mean is negative, Phi(-mu/sigma)"
    # This specific directive suggests a different path. Let's look at the "INSPIRATIONS".
    # Inspiration 2: "Replace the static LCB heuristic with a dynamic acquisition that explicitly blends EI and UCB"
    
    # Let's try a different structural change: 
    # **Posterior Sampling-based Acquisition (Thompson Sampling approximation)**
    # Instead of deterministic EI/UCB, we can estimate the expected improvement
    # by considering the distribution of the optimal value.
    # However, we only have mu and sigma.
    
    # Let's try the "Zero-Crossing" idea from the directive more seriously,
    # but adapted for minimization where the target is 0.
    # Score = P(Y < 0) + EI(Y < 0)
    # This directly targets the theoretical minimum.
    
    z_zero = (0.0 - mu) / s
    prob_zero = _norm_cdf_vec(z_zero)
    
    # EI towards 0
    ei_zero = (0.0 - mu) * _norm_cdf_vec(z_zero) + s * _norm_pdf_vec(z_zero)
    ei_zero = np.maximum(ei_zero, 0.0)
    
    # Combine: Weight the probability of being below zero heavily
    # This creates a strong gradient towards x=0.
    # As mu approaches 0, prob_zero approaches 0.5.
    # As sigma decreases, the transition becomes sharper.
    
    # To avoid getting stuck if we miss 0, we should also consider improvement over f_best.
    # Let's blend "Target 0" and "Improve f_best".
    
    z_fb = (fb - mu) / s
    prob_fb = _norm_cdf_vec(z_fb)
    ei_fb = (fb - mu) * _norm_cdf_vec(z_fb) + s * _norm_pdf_vec(z_fb)
    ei_fb = np.maximum(ei_fb, 0.0)
    
    # Adaptive weight: 
    # If f_best is close to 0, focus on 0.
    # If f_best is far from 0, focus on improving f_best.
    # distance = abs(fb)
    # weight_0 = 1.0 / (1.0 + abs(fb))  # High weight on 0 when fb is small
    # weight_fb = 1.0 - weight_0
    
    # Let's use a sigmoidal weight based on fb
    # If fb > 1.0, weight_fb dominates.
    # If fb < 0.1, weight_0 dominates.
    w_0 = 1.0 / (1.0 + np.exp(10.0 * (fb - 0.5))) # Roughly 0.5 at fb=0.5, 1.0 at fb=0
    
    # Actually, simpler:
    # Score = w_0 * (prob_zero + ei_zero) + (1-w_0) * (prob_fb + ei_fb)
    
    # But wait, the directive says: "Model the acquisition as a probabilistic 'zero-crossing' likelihood... Phi(-mu/sigma)"
    # Let's implement a hybrid that strongly favors the zero-crossing but retains exploration.
    
    # Structural Change: 
    # Use **Probability of Improvement (PI)** towards 0, boosted by **Uncertainty**.
    # This is different from EI because PI is bounded [0,1] and doesn't have the "tail" of EI.
    # It creates a steeper gradient near the decision boundary.
    
    # Let's try: Score = PI_0 * (1 + sigma) 
    # This rewards candidates that are likely to be < 0 AND have high uncertainty (exploration).
    # As we converge, sigma drops, and the score drops for high-x candidates.
    
    # However, if we are far from 0, PI_0 is ~0. We need to move towards 0.
    # A better proxy for "moving towards 0" is -mu.
    # Score = PI_0 * K + (-mu) * (1 - PI_0) ?
    
    # Let's look at the "INSPIRATION 1" and "2".
    # Inspiration 2: Blend EI and UCB dynamically.
    
    # Let's implement a **Dynamic UCB-EI Blend** where the blend coefficient is 
    # determined by the ratio of current best to the expected minimum (0).
    
    # 1. Calculate EI towards f_best
    s_safe = np.maximum(sigma, 1e-8)
    z_fb = (fb - mu) / s_safe
    ei_fb = (fb - mu) * _norm_cdf_vec(z_fb) + s_safe * _norm_pdf_vec(z_fb)
    ei_fb = np.maximum(ei_fb, 0.0)
    
    # 2. Calculate UCB (for minimization, we want low mu, high sigma)
    # Standard UCB: -mu + beta * sigma
    # Let's use a beta that adapts.
    # If we are early (fb is large), we need more exploration.
    # If we are late (fb is small), we need exploitation.
    
    # Metric for "progress": fb itself.
    # If fb is large, we are far. 
    # Let's define a progress variable p in [0, 1].
    # p = 1.0 / (1.0 + fb)  # High p when fb is small (close to min)
    
    # Exploration coefficient beta
    # When p is low (early), beta should be high.
    # When p is high (late), beta should be low.
    # beta = 2.0 * (1.0 - p) + 0.1  # Range [0.1, 2.1]
    
    p = 1.0 / (1.0 + max(fb, 0.0))
    beta = 0.5 + 1.5 * (1.0 - p)
    
    ucb = -mu + beta * s_safe
    
    # 3. Blend
    # We want to switch from UCB (exploration) to EI (exploitation) as we progress.
    # weight_ei = p
    # weight_ucb = 1.0 - p
    
    # But EI and UCB have different scales.
    # Let's normalize them? No, keep it simple.
    # If fb is large, EI is large. UCB is large.
    # If fb is small, EI is small. UCB is dominated by -mu.
    
    # Actually, a common structural change is **Knowledge Gradient (KG)**.
    # KG estimates the expected value of perfect information.
    # Approximation: EI / sigma. This rewards confident improvements.
    
    # Let's try **Confidence-Weighted EI**.
    # Score = EI * (1 / (1 + sigma))
    # This penalizes high uncertainty. Good for exploitation.
    # But we need exploration early.
    
    # Let's go with the **Dynamic UCB-EI Blend** as it's a clear structural shift from static portfolio.
    
    score_ei = ei_fb
    score_ucb = ucb
    
    # To blend, we need to handle scales.
    # Let's just add them with adaptive weights.
    # If p is high (close to min), trust EI.
    # If p is low (far from min), trust UCB.
    
    w_ei = p
    w_ucb = 1.0 - p
    
    # We must ensure scores are finite.
    # Also, UCB can be negative if mu is very negative (which is good, but we want "higher is better" for selection).
    # Wait, the task says "higher meaning evaluate next".
    # For minimization, we want to select points with LOW f(x).
    # So our acquisition score should be HIGH for good points.
    # EI is high for good points.
    # UCB (-mu + beta*sigma) is high for low mu (good) and high sigma (exploration).
    # So both are oriented correctly: High score = Good candidate.
    
    final_score = w_ei * score_ei + w_ucb * score_ucb
    
    # Safety: ensure finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple LCB if running out of time
        s = np.maximum(sigma, 1e-8)
        return -mu + 1.0 * s

    return _ei_dynamic(mu, sigma, fb, 1.0)
# EVOLVE-BLOCK-END
