import numpy as np
import time
from scipy.stats import norm, t as scipy_t

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using robust expected improvement with Student-t
    distribution for uncertainty-weighted cost-aware selection.
    
    Combines the structural idea of EI per unit cost with a refinement that uses
    a Student-t based improvement metric to account for model uncertainty,
    plus a non-linear improvement reward for better exploration/exploitation balance.
    """
    # Track time for safety margin
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    # Standard Normal CDF and PDF
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Student-t based robust improvement metric
    # Use degrees of freedom that adapt to the scale of uncertainty
    # Lower DOF = heavier tails = more robust to outliers in the improvement
    # We use a fixed moderate DOF (5) for robustness while maintaining sensitivity
    dof = 5.0
    
    # For Student-t, E[max(Y, 0)] where Y ~ t_dof scaled appropriately
    # We approximate the Student-t expected improvement using the relationship:
    # If Z ~ N(0,1), then T = Z / sqrt(V/dof) where V ~ chi2_dof
    # For the expected improvement under Student-t, we use a numerical approximation
    # via the fact that t-distribution CDF/PDF can be computed directly
    
    # Student-t CDF and PDF at z
    t_cdf_val = scipy_t.cdf(z_clipped, dof)
    t_pdf_val = scipy_t.pdf(z_clipped, dof)
    
    # Scale factor for Student-t: variance of t_dof is dof/(dof-2) for dof > 2
    t_scale = np.sqrt(dof / (dof - 2.0))
    
    # Student-t Expected Improvement (robust version)
    # The improvement is (f_best - mu) * P(f < f_best) + E[(f_best - f) * 1(f < f_best)]
    # For Student-t, this becomes:
    st_ei = (f_best - mu) * t_cdf_val + sigma * t_scale * t_pdf_val
    
    # Non-linear improvement reward: emphasize larger improvements more
    # Use a power transformation on the improvement magnitude
    improvement_magnitude = np.maximum(f_best - mu, 0)
    non_linear_reward = np.power(improvement_magnitude, 0.8) * np.exp(-0.1 * sigma_safe)
    
    # Combine metrics with adaptive weights
    # Weight 1: Standard EI per cost (proven baseline)
    score_ei = ei / cost_safe
    
    # Weight 2: Student-t robust EI per cost
    score_st = st_ei / cost_safe
    
    # Weight 3: Non-linear reward per cost
    score_nl = non_linear_reward / cost_safe
    
    # Adaptive combination: use more robust metrics when uncertainty is high
    # Normalize sigma to [0, 1] for adaptive weighting
    sigma_max = np.max(sigma_safe) + 1e-10
    sigma_norm = sigma_safe / sigma_max
    
    # When sigma is high, trust the robust Student-t metric more
    # When sigma is low, the standard EI is more reliable
    w_st = 0.3 + 0.4 * sigma_norm  # Weight for Student-t: 0.3 to 0.7
    w_ei = 0.5 - 0.3 * sigma_norm  # Weight for standard EI: 0.5 to 0.2
    w_nl = 0.2                     # Fixed weight for non-linear reward
    
    # Normalize weights to sum to 1
    weight_sum = w_st + w_ei + w_nl
    w_st /= weight_sum
    w_ei /= weight_sum
    w_nl /= weight_sum
    
    scores = w_ei * score_ei + w_st * score_st + w_nl * score_nl
    
    # Check deadline
    if time.time() > deadline:
        # Return the simplest and fastest score as fallback
        return score_ei
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
