"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

def _norm_cdf(x):
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Structural change: Uses Expected Improvement (EI) divided by cost, which
    directly optimizes expected improvement per unit budget. EI is defined as:
    
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    Key structural differences from quantile-based approach:
    1. Uses f_best as a dynamic reference point (adapts as search progresses)
    2. Has a fundamentally different mathematical form (involves normal CDF/PDF)
    3. Naturally balances exploration and exploitation through the EI formulation
    4. Cost division is applied to the full EI, not to a quantile
    
    This should better handle the multi-instance setting where minimizers are
    shifted, as EI naturally explores regions with high uncertainty (high sigma)
    while exploiting regions where the GP predicts values below f_best.
    """
    # Deadline guard: if we're running out of time, return simple exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, use the standard form:
    # EI = sigma * [z * Phi(z) + phi(z)]  when z > 0
    # EI = 0 when z <= 0 (no improvement expected)
    
    phi_z = _norm_pdf(z)
    Phi_z = _norm_cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (clip small negative values from numerical error)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
