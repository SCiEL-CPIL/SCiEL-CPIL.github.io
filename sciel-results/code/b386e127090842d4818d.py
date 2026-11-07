import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and Probability of Improvement (PI)
    with an adaptive exploration weight based on the current best value.
    
    This is a structurally different approach from the current Adaptive UCB:
    - UCB is linear in sigma and can be dominated by the -mu term or the beta*sigma term in extreme cases.
    - EI is non-linear and bounded, providing a more balanced exploration-exploitation trade-off.
    - PI provides a strong exploration signal in regions of high uncertainty even when the mean is poor.
    
    We combine these: Score = w * EI + (1-w) * PI * sigma_scale
    where w is adaptive based on f_best.
    
    Additionally, we apply a "soft" penalty for points where the predicted improvement is likely to be negligible
    but the uncertainty is high, to avoid wasting evaluations on "noise" regions far from the basin of attraction.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin for 300s limit
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Normalize EI and PI to comparable scales?
    # EI can be large if f_best is large. PI is in [0,1].
    # Let's use a weighted sum where the weight shifts from exploration to exploitation.
    
    # Adaptive weight:
    # If f_best is high (early), we want to explore. PI is a good exploration metric.
    # If f_best is low (late), we want to exploit. EI is a good exploitation metric.
    # However, EI also has an exploration component (the sigma*phi(z) term).
    
    # Let's try a different structural change: **Knowledge Gradient (KG) Approximation via PI-EI blend**
    # Actually, let's look at **Probability of Best (PoB)** or **Thompson Sampling** proxy.
    # But we must return scores.
    
    # Let's try **Maximizing the Expected Minimum Improvement** which is related to KG.
    # A simpler, robust alternative to UCB is **Expected Hypervolume Improvement (EHVI)**
    # but for single objective, EHVI reduces to a form of EI.
    
    # Let's try **Lower Confidence Bound (LCB)**? No, that's just -UCB.
    
    # Let's try **Bayesian Optimization with a Repulsive Term** is not possible without x.
    
    # Let's try **Probability of Improvement (PI)** with a **Sigma-Boost**.
    # Score = PI * (1 + alpha * sigma)
    # This encourages high uncertainty regions (exploration) but weights them by the probability of improvement.
    # If mu is much worse than f_best, PI is ~0, so score is ~0.
    # If mu is close to f_best and sigma is high, PI is ~0.5, score is boosted.
    # If mu is much better than f_best, PI is ~1, score is ~1 + alpha*sigma.
    
    # This is different from UCB because UCB = -mu + beta*sigma.
    # If mu is very bad (large), UCB is very negative.
    # If mu is very good (small), UCB is very positive.
    # PI-based score is bounded by the probability, which might be too conservative for exploitation?
    # No, if mu is much better than f_best, PI is 1. We want to pick it.
    # But if mu is 1.0 and f_best is 0.1, PI is small. EI is 0.
    # We want to find points with low mu.
    
    # Let's combine EI and a "Confidence" term.
    # Actually, the current Adaptive UCB scored 3.0.
    # Inspiration 1 (UCB) scored 2.15.
    # Inspiration 2 (Unknown) scored 1.5.
    # Inspiration 3 (Thompson Sampling) scored 1.35.
    
    # The Adaptive UCB is the best so far.
    # How to improve it structurally?
    # Maybe the linear scaling of beta with f_best is not optimal.
    # What if we use a **Non-Linear Acquisition Function** like **Expected Improvement** but with a **Lookahead**?
    
    # Let's try **Expected Improvement (EI)** but with a **Boost for Uncertainty** that is not linear.
    # Or **Probability of Improvement (PI)** with a **Mean-Shift**.
    
    # Let's try **Score = EI + gamma * sigma * (1 - PI)**
    # This adds an exploration bonus that is high when PI is low (i.e., when we are uncertain about improvement).
    
    # Let's calibrate gamma.
    # If f_best is high, we want high gamma.
    # gamma = 0.5 * f_best
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple UCB if taking too long (shouldn't happen for vectorized ops)
        beta = 0.5 + 2.0 * np.clip(f_best, 0.0, 10.0)
        return -mu + beta * sigma_safe
    
    gamma = 0.5 * np.clip(f_best, 0.0, 10.0)
    
    # Exploration bonus: High when sigma is high and PI is low (uncertain)
    # If PI is high, we are confident it's good, so we don't need as much bonus.
    # If PI is low, it might be bad or just uncertain. The bonus encourages looking at uncertain spots.
    exploration_bonus = gamma * sigma_safe * (1.0 - pi)
    
    scores = ei + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
