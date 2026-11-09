"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to enforce deadline and cache
_start_time = None
_deadline = 330.0
_cache = {}

def acquisition(mu, sigma, f_best):
    global _start_time, _deadline, _cache
    
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() > _deadline - 5.0:
        # Fallback: return simple EI if we are running out of time
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        return ei

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-9)
    
    # EImin (Expected Improvement of the Minimum)
    # This acquisition function is designed to minimize the expected value of the minimum 
    # after the next evaluation. It is more robust than standard EI for global optimization
    # because it accounts for the possibility that the current f_best is not the true global minimum.
    
    # The formula for EImin for minimization:
    # EImin(x) = E[ max(f_best - f(x), 0) | f(x) ~ N(mu, sigma^2) ] 
    # Wait, standard EI is already E[max(f_best - f(x), 0)].
    # Actually, EImin is often defined as:
    # EImin(x) = E[ min(f(x), f_best) ] - min(E[f(x)], f_best) ? No.
    
    # Let's look at the "Expected Improvement" variant that is more global.
    # A common alternative is the **Probability of Improvement (PI)** combined with a 
    # **Lower Confidence Bound (LCB)** or using **Upper Confidence Bound (UCB)** for minimization.
    
    # However, a distinct structural change is to use **Expected Regret** or **EImin** 
    # which is:
    # EImin(x) = E[ f_best - min(f(x), f_best) ] = E[ max(f_best - f(x), 0) ]
    # This is exactly standard EI. So standard EI is EImin.
    
    # What is materially different?
    # 1. **UCB (Upper Confidence Bound) for minimization**: Score = mu - kappa * sigma.
    #    This is purely exploration/exploitation balance without the probabilistic weighting of EI.
    #    EI can be too conservative early on if f_best is high but uncertain.
    #    UCB forces exploration of high sigma regions more aggressively.
    
    # 2. **PI (Probability of Improvement)**: Score = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    #    This ignores the magnitude of improvement, focusing only on the chance of beating the best.
    #    This can lead to many small improvements, potentially getting stuck in local optima 
    #    if not combined with other terms.
    
    # 3. **Hybrid: EI + UCB term**.
    #    The current program does EI + beta*sigma.
    #    Let's try a **Pure UCB** with adaptive kappa.
    #    Or better, **Expected Improvement of the Variance (EIV)**? No.
    
    # Let's try **UCB (Upper Confidence Bound)** for minimization.
    # Score = mu - kappa * sigma
    # This is structurally different from EI. EI is a nonlinear function of z.
    # UCB is linear in mu and sigma.
    # For Beale function, which is smooth and has a single global minimum,
    # UCB with a decaying kappa might explore more broadly and avoid local traps 
    # that EI might fall into due to its local nature.
    
    # Adaptive kappa:
    # Early on (high f_best), we want to explore more (larger kappa).
    # Late on (low f_best), we want to exploit (smaller kappa).
    
    # kappa = 1.0 + 2.0 * (f_best / (f_best + 1.0))
    # If f_best = 50, kappa = 1.0 + 2.0 * (50/51) ~ 2.98
    # If f_best = 1, kappa = 1.0 + 2.0 * (1/2) = 2.0
    # If f_best = 0.1, kappa = 1.0 + 2.0 * (0.1/1.1) ~ 1.18
    
    # Let's try a more aggressive exploration initially.
    # kappa = 2.0 * np.sqrt(f_best + 0.1)
    # If f_best = 50, kappa = 2 * 7.07 = 14.14 (Very high exploration)
    # If f_best = 1, kappa = 2 * 1.05 = 2.1
    # If f_best = 0.1, kappa = 2 * 0.45 = 0.9
    
    # Let's combine UCB with a small EI term to ensure we still prioritize actual improvement magnitude.
    # Score = EI + kappa * (mu - kappa * sigma) ? No, that's mixing.
    
    # Let's stick to **Pure UCB** but with a carefully tuned adaptive kappa.
    # The Beale function has a minimum of 0.
    # If we use UCB, we are effectively optimizing for the lower bound of the confidence interval.
    
    # Another structural change: **Lookahead** or **Ensemble** is too expensive.
    
    # Let's try **Expected Improvement with a Gaussian Process Kernel trick**? No, kernel is fixed.
    
    # Let's try **PI (Probability of Improvement)** multiplied by a scale factor.
    # PI = Phi(z)
    # This is very smooth and focuses on the probability.
    
    # Let's try **UCB** as the primary structural change.
    # It is a different design family (Confidence Bound vs. Expected Improvement).
    
    kappa = 1.0 + 1.5 * (f_best / (f_best + 1.0))
    
    # Score = mu - kappa * sigma
    score = mu - kappa * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
