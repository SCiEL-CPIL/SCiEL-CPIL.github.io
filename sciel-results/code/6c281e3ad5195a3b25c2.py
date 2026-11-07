import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5 seconds margin before 330s

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: simple PI / cost
        z = (f_best - mu) / (sigma + 1e-9)
        pi = norm.cdf(z)
        return pi / cost

    # Adaptive lambda
    # If f_best is large, lambda should be small to avoid saturation
    # If f_best is small, lambda can be larger
    lam = 1.0 / (1.0 + abs(f_best))
    
    m = f_best - mu
    s = sigma + 1e-9
    
    # Calculate P(Y > 0)
    prob_improve = norm.cdf(m / s)
    
    # Calculate E[exp(-lam * Y) * I(Y > 0)]
    # Term: exp(0.5 * lam^2 * s^2 - lam * m) * Phi((m - lam * s^2) / s)
    
    # Numerical stability
    # If s is very small, m/s is large.
    # If lam * s^2 is large, the argument to Phi is large negative.
    
    arg1 = m / s
    arg2 = (m - lam * s**2) / s
    
    # Compute log of the second term to avoid overflow/underflow
    # log_term = 0.5 * lam**2 * s**2 - lam * m + log(Phi(arg2))
    # If Phi(arg2) is 0, log is -inf.
    
    # Using scipy.special for log pdf/cdf? norm.logcdf
    log_phi2 = norm.logcdf(arg2)
    log_exp_part = 0.5 * lam**2 * s**2 - lam * m
    
    # If log_exp_part + log_phi2 is very large, exp will overflow.
    # But prob_improve is bounded [0,1].
    # The second term is part of a subtraction.
    
    # Let's try direct calculation with clipping for safety
    # If s is large, lam*s^2 is large, arg2 is very negative, Phi(arg2) ~ 0.
    # exp part might be large.
    
    # Safe approach:
    # If arg2 < -10, Phi(arg2) is effectively 0.
    # If arg2 > 10, Phi(arg2) is 1.
    
    phi2 = norm.cdf(arg2)
    
    # Calculate the exponential part
    exp_arg = 0.5 * lam**2 * s**2 - lam * m
    
    # Clip exp_arg to prevent overflow
    # Max exp arg for double is ~700
    exp_arg_clipped = np.clip(exp_arg, -700, 700)
    exp_part = np.exp(exp_arg_clipped)
    
    integral_term = exp_part * phi2
    
    # E[U] = P(Y>0) - Integral
    e_u = prob_improve - integral_term
    
    # Ensure non-negative (numerical errors)
    e_u = np.maximum(e_u, 0.0)
    
    scores = e_u / cost
    
    return scores
