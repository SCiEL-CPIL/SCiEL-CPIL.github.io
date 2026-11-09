"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a hybrid of Expected Improvement and Upper Confidence Bound,
    with adaptive exploration to balance exploitation and exploration.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Sanitize sigma: ensure positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Sanitize mu: replace non-finite with large penalty
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Compute z-score for EI
    z = (f_best - mu_clean) / sigma_safe
    z = np.clip(z, -10, 10)  # Prevent overflow in norm.cdf/pdf
    
    # Expected Improvement
    ei = (f_best - mu_clean) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI should be non-negative
    
    # Upper Confidence Bound (UCB) component for minimization
    # We want to balance exploitation (low mu) and exploration (high sigma)
    # Standard UCB for minimization: mu - kappa * sigma
    # But we need a score where HIGHER is better.
    # So we can use: -mu + kappa * sigma (higher if mu is low or sigma is high)
    
    # Adaptive kappa: start higher, potentially decay if f_best is good
    # For Michalewicz 2D, the landscape has many local minima.
    # A fixed kappa around 1.0-2.0 often works well for early/mid exploration.
    # Let's use a slightly aggressive exploration factor.
    kappa = 1.5
    
    ucb_score = -mu_clean + kappa * sigma_safe
    
    # Blend EI and UCB
    # EI is bounded by (f_best - min(mu)), which shrinks as f_best improves.
    # UCB scales with the range of mu and sigma.
    
    # Simple approach: Weighted sum.
    # If EI is 0 everywhere (mu >= f_best everywhere), we fall back to UCB.
    
    # Let's try a direct combination:
    # Score = alpha * normalized_ei + (1-alpha) * normalized_ucb
    
    # However, normalization can be tricky if variances are tiny.
    # Alternative: Use EI when it's significant, otherwise UCB.
    
    # Let's look at the structure.
    # EI = 0 if mu >= f_best.
    # UCB is always active.
    
    # A common robust strategy is the "Noisy EI" or just EI with a floor.
    # But given the task constraints and the need to jump the score,
    # a pure EI might be too greedy if the GP is overconfident in bad regions.
    
    # Let's try a hybrid:
    # score = ei + beta * sigma
    # This encourages exploring high-uncertainty areas even if current mu is not best.
    # beta controls the exploration trade-off.
    
    # Let's set beta to a small value relative to the scale of sigma.
    # Since targets are standardized in the GP training, sigma is in standardized units?
    # Wait, the contract says: "mu and sigma ... are in objective units."
    # So sigma is in the same scale as f_best and mu.
    
    # If f_best is around -1.0 and the range is [-1.8, 0.6], sigma might be ~0.1-0.5.
    # EI can be up to ~0.5.
    
    # Let's use a weighted sum:
    # score = ei + 0.5 * sigma
    # This adds a bonus for uncertainty.
    
    # Actually, a better approach for Michalewicz is often the "Knowledge Gradient" or just EI.
    # But EI can get stuck.
    
    # Let's try the "Lower Confidence Bound" (LCB) for minimization, which is equivalent to maximizing:
    # -mu + kappa * sigma
    # This is known to be very effective for multimodal functions.
    
    # Let's combine them:
    # We will use EI as the base, but add a UCB-like exploration term.
    
    # To make it scale-invariant and robust:
    # 1. Calculate EI.
    # 2. Calculate UCB = -mu + kappa * sigma.
    # 3. Return a weighted sum.
    
    # Let's pick weights that favor EI when it's large, and UCB when it's not.
    # Or simply: return ei + 0.1 * sigma?
    
    # Let's look at typical BO performance. EI is the standard.
    # The current seed uses EI. Score is 1.0.
    # To improve, we need better exploration or a different trade-off.
    
    # Let's try a higher kappa in a UCB-like manner blended with EI.
    
    # Score = EI + 0.2 * Sigma
    # This is a simple "Noisy EI" variant.
    
    # Let's refine:
    # If we just return EI, it might miss distant promising regions.
    # UCB explores more.
    
    # Let's try:
    # score = ei + 0.3 * sigma_safe
    
    # Clip scores to finite values
    score = ei + 0.3 * sigma_safe
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
