import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
# Standard normal CDF and PDF
def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(norm.cdf)(x))

def _norm_pdf(x):
    return np.vectorize(norm.pdf)(x)

# Actually, let's just use the scipy functions directly but efficiently.
# We need to compute the Expected Improvement (EI) for minimization.
# EI(x) = E[max(f_best - f(x), 0)]
# Let Z = (f(x) - f_best) / sigma(x)
# Then EI = (f_best - mu) * Phi(-(f_best - mu)/sigma) + sigma * phi(-(f_best - mu)/sigma)
# Note: for minimization, we want to maximize EI.
# If sigma is very small, we should avoid division by zero.

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Clip sigma to avoid division by zero or very large values
    # The problem states sigma is standard deviation in objective units.
    # A small epsilon to prevent numerical issues.
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the Z-score for each candidate
    # Z = (f_best - mu) / sigma
    # If mu < f_best, Z is positive. The probability that f(x) < f_best is Phi(-Z) = Phi((mu - f_best)/sigma).
    # Let's use the standard formula for EI in minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Wait, let's verify.
    # Let Y = f(x) ~ N(mu, sigma^2).
    # We want E[max(f_best - Y, 0)].
    # This is the same as E[max(Y - f_best, 0)] but with sign flipped? No.
    # Let D = f_best - Y. We want E[max(D, 0)].
    # D ~ N(f_best - mu, sigma^2).
    # Let m = f_best - mu, s = sigma.
    # E[max(D, 0)] = m * Phi(m/s) + s * phi(m/s).
    # Where Phi is standard normal CDF, phi is standard normal PDF.
    
    m = f_best - mu
    s = sigma_safe
    z = m / s
    
    # Compute Phi(z) and phi(z)
    # Using scipy.stats.norm for vectorized operations
    # norm.cdf and norm.pdf are vectorized.
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = m * Phi_z + s * phi_z
    
    # The acquisition function should return scores where higher is better.
    # So we return EI directly.
    
    # Deadline guard: if we are taking too long, return a simple fallback?
    # The computation above is O(N) and very fast for N=2000.
    # It should take < 1ms. So the deadline guard is likely unnecessary for this specific calculation.
    # However, to be safe and adhere to the directive:
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to UCB if EI calculation was unexpectedly slow (unlikely)
        return -mu + 1.0 * sigma_safe
    
    return ei
