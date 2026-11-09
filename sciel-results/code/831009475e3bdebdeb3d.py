"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus
    that penalizes regions where the posterior mean is already better than f_best
    (to avoid redundant evaluations) and rewards high uncertainty in unexplored areas.
    
    This is a modified EI with a 'lower bound' correction to handle the case where
    the GP posterior might predict values lower than f_best due to noise or
    overconfidence, which can lead to poor exploration.
    
    We use a standard EI but add a small penalty proportional to (f_best - mu) when
    mu < f_best to discourage selecting points that are already 'good' according to
    the mean, forcing exploration of more uncertain regions.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward points with high sigma, but dampen if mu is already very good
    # The term (mu - f_best) is positive when mu > f_best (worse than best), negative when mu < f_best.
    # We want to boost exploration in regions with high uncertainty.
    # A simple approach: add a small fraction of sigma.
    # However, pure EI already does this.
    # Let's try a different angle: UCB (Upper Confidence Bound)
    # UCB = mu + beta * sigma
    # But we are minimizing, so we want low mu and high sigma.
    # Actually, for minimization, UCB is mu + beta * sigma (we want to maximize this score to find low values? No.)
    # For minimization, we usually use LCB (Lower Confidence Bound) for the next point?
    # No, the acquisition function returns a score, higher is better.
    # If we want to find low f, we want high probability of f < f_best.
    # EI is the standard choice.
    
    # Let's try a hybrid: 0.5 * EI + 0.5 * UCB (where UCB for minimization is -mu + beta*sigma? No.)
    # Let's stick to EI but ensure numerical stability.
    # The current seed is pure EI.
    
    # Alternative: Probability of Improvement (PI)
    # pi = norm.cdf(z)
    # PI is more exploratory than EI? Not necessarily.
    
    # Let's try a "Noisy" EI or a tuned EI.
    # One common issue with EI is that it can be too greedy.
    # Adding a small exploration term based on sigma can help.
    
    # Let's use a combination: EI + lambda * sigma
    # This encourages exploration of high variance regions even if EI is low.
    # However, EI already includes sigma.
    
    # Let's try the "Expected Hypervolume Improvement" approximation? Too complex.
    
    # Let's try a simple modification:
    # If the GP is very confident (low sigma) and mu is close to f_best, EI is small.
    # If we add a term that rewards sigma, we explore more.
    
    # Let's try: score = EI + 0.1 * sigma
    # This is a heuristic.
    
    # Actually, a more robust structural change is to use the **Upper Confidence Bound (UCB)**
    # for minimization: 
    # We want to maximize the score.
    # For minimization, the "value" of a point is its function value.
    # We want to find low values.
    # UCB for minimization: mu + beta * sigma? 
    # If we pick the point with the highest (mu + beta * sigma), we might pick high mu points.
    # That's wrong.
    # For minimization, we want to pick points that *could* be low.
    # So we want to maximize the probability of being low, or the expected improvement.
    
    # Let's reconsider the **Lower Confidence Bound (LCB)** for minimization.
    # LCB = mu - beta * sigma.
    # We want to minimize LCB.
    # So the score should be -LCB = -mu + beta * sigma.
    # This rewards low mu and high sigma.
    # This is a very different acquisition function than EI.
    # EI is local, UCB/LCB is global.
    
    # Let's try LCB-based score:
    # score = -mu + beta * sigma
    
    # We need to choose beta.
    # A common choice is beta = 2 * log(n) or similar, but we don't have n here.
    # We can use a fixed beta, e.g., 1.0 or 2.0.
    # Since the domain is normalized to [0,1]^2 and length scale is 0.2, the function is relatively smooth.
    # Let's try beta = 1.0.
    
    # However, LCB can be very aggressive in exploration.
    # A blend of EI and LCB might be good.
    
    # Let's try: score = 0.5 * EI + 0.5 * (-mu + sigma)
    # This balances exploitation (EI) and exploration (LCB).
    
    # Let's implement this hybrid.
    
    # Normalize EI and LCB to have similar scales?
    # EI is in objective units. -mu + sigma is in objective units.
    # They are comparable.
    
    # Let's try this hybrid.
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate LCB score (to be maximized)
    lcb_score = -mu + 1.0 * sigma_safe
    
    # Hybrid
    score = 0.5 * ei + 0.5 * lcb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
