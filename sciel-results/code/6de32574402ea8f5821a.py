import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 275.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a robust, quantile-normalized Thompson Sampling heuristic.
    
    Mechanism:
    1. Normalizes both mu and sigma by their respective medians to remove scale dependency.
    2. Estimates a global exploration scale using the 90th percentile of normalized sigma,
       which is robust to outliers compared to max or std.
    3. Combines exploitation (-normalized_mu) and exploration (robust_sigma) with a 
       dynamic weight that decays as f_best improves.
    """
    # Deadline guard: Stop trying to optimize if we are close to the time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])

    # 1. Robust Normalization
    # Use median for centering/scaling to be robust against outliers in the GP posterior
    mu_med = np.median(mu)
    sig_med = np.median(sigma)
    
    # Avoid division by zero
    if mu_med == 0:
        mu_med = 1e-9
    if sig_med == 0:
        sig_med = 1e-9

    mu_norm = (mu - mu_med) / abs(mu_med)
    sig_norm = sigma / sig_med
    
    # Clip extreme values in normalization to prevent numerical issues
    mu_norm = np.clip(mu_norm, -100, 100)
    sig_norm = np.clip(sig_norm, 0, 100)

    # 2. Dynamic Exploration Weight
    # Calculate a robust "typical" uncertainty level across all candidates
    # Using the 75th percentile of sigma_norm gives a sense of the "upper-middle" uncertainty
    # without being as extreme as the max.
    sigma_typical = np.percentile(sig_norm, 75)
    
    if sigma_typical < 1e-6:
        sigma_typical = 1e-6

    # The exploration weight beta should be high when f_best is large (early game)
    # and low when f_best is small (late game).
    # We use a reciprocal-like decay to keep exploration active but diminishing.
    # lambda_ts = 1.0 / (1.0 + f_best) is one option, but we want it to be more aggressive early.
    # Let's use: beta = 1.0 * (1.0 / (1.0 + f_best**0.5))
    # This ensures beta is ~1.0 when f_best is near 0? No, that's backwards.
    # If f_best is large (e.g., 5), beta should be large.
    # If f_best is small (e.g., 0.1), beta should be small.
    
    # Corrected logic:
    # Early: f_best is high (e.g., 5). We want to explore.
    # Late: f_best is low (e.g., 0.01). We want to exploit.
    
    # Let's use a weight that scales with the magnitude of the best value found so far.
    # Since Ackley's min is 0, f_best is always positive.
    # If f_best is 5, beta = 5. If f_best is 0.1, beta = 0.1.
    # This might be too strong. Let's dampen it.
    
    beta = np.sqrt(f_best)
    
    # 3. Score Calculation
    # We want to maximize the score.
    # Exploitation term: -mu_norm (we prefer low mu, i.e., negative normalized mean relative to median)
    # Exploration term: beta * sig_norm (we prefer high uncertainty)
    
    # To balance the scales, we normalize the exploration term by the typical uncertainty
    # so that a point with 2x typical uncertainty gets 2x the exploration bonus relative to typical.
    sig_relative = sig_norm / sigma_typical
    
    # Final Score
    # We combine the normalized mean and the relative uncertainty.
    score = -mu_norm + beta * sig_relative
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
