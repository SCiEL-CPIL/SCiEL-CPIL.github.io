"""Cost-aware acquisition function using Student-t Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # Safety margin: kill at 330s, we stop at 290s


def _student_t_ei(mu, sigma, f_best, nu=3.0):
    """
    Expected Improvement using a Student-t distribution.
    For a Student-t with df=nu, the EI is:
    EI = (f_best - mu) * P(X < f_best) + sigma * E[(f_best - X) | X < f_best] * P(X < f_best)
    
    Where X ~ t_nu(mu/scale, sigma/scale)
    
    Actually, let's use the standard formula for Student-t EI.
    If Y ~ t_nu(0,1), then X = mu + sigma * Y
    EI = E[max(f_best - X, 0)]
       = E[max(f_best - mu - sigma*Y, 0)]
       = (f_best - mu) * P(Y < (f_best - mu)/sigma) + sigma * E[max((f_best-mu)/sigma - Y, 0)]
    
    For standard normal: E[max(z - Y, 0)] = z * phi(z) + phi(z) ... wait, that's not right.
    For standard normal Y: E[max(a - Y, 0)] = a * Phi(a) + phi(a)
    So EI = (f_best - mu) * Phi(a) + sigma * [a * Phi(a) + phi(a)] where a = (f_best-mu)/sigma
    EI = (f_best - mu) * Phi(a) + sigma * a * Phi(a) + sigma * phi(a)
    EI = (f_best - mu) * Phi(a) + (f_best - mu) * Phi(a) + sigma * phi(a)
    EI = 2*(f_best - mu) * Phi(a) + sigma * phi(a)  -- this is wrong, let me re-derive.
    
    Let a = (f_best - mu) / sigma
    EI = E[max(f_best - mu - sigma*Y, 0)]
       = E[max(sigma*(a - Y), 0)]
       = sigma * E[max(a - Y, 0)]
    
    For Y ~ N(0,1): E[max(a - Y, 0)] = a * Phi(a) + phi(a)
    So EI = sigma * (a * Phi(a) + phi(a)) = (f_best - mu) * Phi(a) + sigma * phi(a)
    
    For Y ~ t_nu(0,1):
    E[max(a - Y, 0)] = a * F_t(a) + f_t(a) * I(a)
    where F_t is the CDF of t_nu, f_t is the PDF of t_nu, and I(a) involves an integral.
    
    Actually, there's a known result. For Student-t with nu degrees of freedom:
    E[max(a - Y, 0)] where Y ~ t_nu(0,1)
    = a * F_t(a) + f_t(a) * (nu + a^2) / (nu - 1)  ... this isn't quite right either.
    
    Let me use a different approach. The PDF of t_nu is:
    f_t(y) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + y^2/nu)^(-(nu+1)/2)
    
    E[max(a - Y, 0)] = integral_{-inf}^{a} (a - y) f_t(y) dy
    = a * F_t(a) - integral_{-inf}^{a} y * f_t(y) dy
    
    For symmetric t distribution, integral_{-inf}^{a} y * f_t(y) dy = -f_t(a) * (nu + a^2) / (nu - 1) * ... 
    
    Actually, let's use the fact that for t_nu:
    d/dy F_t(y) = f_t(y)
    And there's a relationship: y * f_t(y) = -d/dy [f_t(y) * (nu + y^2)/(nu-1) * ... ]
    
    This is getting complex. Let me use a simpler approximation or a different formulation.
    
    Alternative: Use the fact that Student-t can be represented as a scaled normal.
    If Y = Z / sqrt(W/nu) where Z ~ N(0,1), W ~ chi2_nu, independent.
    
    For practical purposes, let's use a numerical integration or a closed-form approximation.
    Actually, there IS a closed form for Student-t EI.
    
    For Y ~ t_nu(0,1):
    E[max(a - Y, 0)] = a * F_t(a) + f_t(a) * (nu + a^2) / (nu - 1)
    
    Wait, I recall that for the standard normal, E[max(a-Y,0)] = a*Phi(a) + phi(a).
    For Student-t, it's: a * F_t(a) + f_t(a) * (nu + a^2) / (nu - 1)
    
    Let me verify: for large nu, t approaches normal, and (nu + a^2)/(nu-1) -> 1, so we get a*F_t(a) + f_t(a) which matches.
    
    So EI = sigma * [a * F_t(a) + f_t(a) * (nu + a^2) / (nu - 1)]
    where a = (f_best - mu) / sigma
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    a = (f_best - mu) / sigma_safe
    
    # Student-t CDF and PDF with nu degrees of freedom
    nu = 3.0
    F_t = student_t.cdf(a, df=nu)
    f_t = student_t.pdf(a, df=nu)
    
    # Handle edge case where nu = 1 (Cauchy), but we use nu=3 so this is fine
    term = a * F_t + f_t * (nu + a**2) / (nu - 1)
    
    ei = sigma_safe * term
    
    # For negative EI (shouldn't happen much), clip to 0
    ei = np.maximum(ei, 0.0)
    
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Student-t Expected Improvement.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: float, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of acquisition scores (higher = more promising)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe default: just EI without cost weighting
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Compute Student-t Expected Improvement
    ei = _student_t_ei(mu, sigma, f_best, nu=3.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
