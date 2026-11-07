import numpy as np
from scipy.stats import t, norm
import time

# Global timer for deadline guard
_start_time = time.time()
TIME_LIMIT = 290.0  # Safety margin from 300s

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Deadline guard: if we've been running too long, fall back to simple UCB
    if time.time() - _start_time > TIME_LIMIT:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)

    sigma = np.maximum(sigma, 1e-9)
    
    # Define improvement for minimization: improvement = f_best - y
    # In standard EI for minimization, we look for E[max(f_best - y, 0)]
    improvement = f_best - mu
    
    # Check if we have any feasible points (improvement > 0)
    # If mu is already better than f_best everywhere, EI is 0, but we might want to explore?
    # Usually f_best is the min observed, so f_best - mu <= 0 for known points, > 0 for unknowns potentially.
    # Actually, for minimization:
    # y ~ N(mu, sigma)
    # Improvement I = max(f_best - y, 0)
    # E[I] = EI
    
    # We use a Student's t-distribution to model the uncertainty of y.
    # Degrees of freedom nu. With 30 evaluations, nu=5 is a common choice for heavy tails.
    nu = 5
    
    # To compute t-EI efficiently, we use Monte Carlo integration with vectorized samples.
    # Since we only have 2000 candidates, we can use a moderate number of samples per candidate.
    # However, doing MC inside the acquisition function for 2000 points might be slow if not vectorized properly.
    # Alternative: Use a closed-form approximation if available, but t-EI doesn't have a simple closed form like Gaussian EI.
    # We will use a small number of MC samples (e.g., 10) vectorized across all 2000 points.
    
    # Let's try 10 samples. 2000 * 10 = 20,000 random numbers. This is very fast in numpy.
    n_samples = 10
    
    # Generate random samples from t-distribution: shape (n_samples, 2000)
    # t.rvs(size=(n_samples, len(mu)), df=nu)
    try:
        t_samples = t.rvs(size=(n_samples, len(mu)), df=nu)
    except Exception:
        # Fallback if t.rvs fails for some reason
        return -mu + 2.0 * sigma

    # y_samples = mu[None, :] + sigma[None, :] * t_samples
    # Note: t.rvs produces standard t (mean 0, var 1 if nu>2).
    # The actual y is mu + sigma * t
    y_samples = mu[None, :] + sigma[None, :] * t_samples
    
    # Improvement samples: I_samples = max(f_best - y_samples, 0)
    i_samples = np.maximum(f_best - y_samples, 0.0)
    
    # Expected t-Improvement
    ei_t = np.mean(i_samples, axis=0)
    
    # Hybrid Score: Combine t-EI with a UCB component to ensure exploration
    # The t-EI already explores, but in the early stages (small f_best relative to landscape),
    # pure EI can be very exploitative if f_best is not yet good.
    # UCB = -mu + beta * sigma
    # Let's blend them.
    # If f_best is very high (initial stage), EI is dominated by uncertainty anyway.
    # As f_best drops, EI becomes more selective.
    
    # Let's use a dynamic beta.
    # beta = 1.0
    ucb_score = -mu + 1.5 * sigma
    
    # We can normalize scores to combine them, or just add them.
    # EI is in objective units. UCB is also in objective units.
    # Let's try a weighted sum.
    # weight_ei = 1.0
    # weight_ucb = 0.1  # Small UCB boost
    
    # Actually, t-EI is already a good balance. Let's stick to t-EI primarily,
    # but add a small UCB term to handle the "all mu > f_best" case where EI=0 but we might still want to move.
    # However, if mu > f_best, it means we've already found better points. EI=0 is correct (no expected improvement).
    # The "exploration" in EI comes from the sigma term in the expectation.
    
    # Let's refine: The standard EI for minimization is:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # For t-distribution, the MC approximation above is valid.
    
    # To differentiate from pure t-EI (which might exist in other programs), let's add a small penalty for points with extremely high sigma
    # that are far from the current best, or just stick to the robust t-EI.
    
    # Let's try a "Clipped" t-EI to avoid numerical instability in far tails, though t.rvs handles it.
    
    # One structural change: Instead of pure t-EI, let's use **Expected Hypervolume Improvement (EHVI)** surrogate?
    # No, EHVI requires more context (other objectives or hypervolume reference).
    
    # Let's stick to the Hybrid: 0.9 * t_EI + 0.1 * UCB
    # This ensures that even if EI is 0 (all predicted means are worse than f_best), we still have a score based on UCB,
    # allowing the search to continue exploring high uncertainty regions.
    
    score = 0.9 * ei_t + 0.1 * ucb_score
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
