"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, erfc, erfcx

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    UCB-style exploration term to encourage more aggressive exploration
    early in the optimization.
    
    Score = EI + kappa * sigma
    where EI is the standard Expected Improvement for minimization.
    """
    # Ensure numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using erfc for numerical stability: Phi(z) = 0.5 * erfc(-z/sqrt(2))
    sqrt2 = np.sqrt(2.0)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # For large positive z, erfc(-z/sqrt(2)) ≈ 1, so Phi(z) ≈ 1
    # For large negative z, erfc(-z/sqrt(2)) ≈ 2, so Phi(z) ≈ 0
    # Use erfcx for better numerical stability in the tails
    # Phi(z) = 0.5 * erfc(-z/sqrt(2))
    # But for very large |z|, we need to be careful
    # Let's use a stable computation
    neg_z_over_sqrt2 = -z / sqrt2
    
    # For z > 0: Phi(z) = 0.5 * erfc(-z/sqrt2) = 0.5 * (1 + erf(z/sqrt2))
    # For z < 0: Phi(z) = 0.5 * erfc(-z/sqrt2)
    # Use erfc for both, but handle the case where the argument is large
    
    # A numerically stable way:
    # Phi(z) = 0.5 * erfc(-z/sqrt(2))
    # When z is very large positive, erfc(-large) ≈ 2, so Phi ≈ 1
    # When z is very large negative, erfc(-negative) = erfc(large) ≈ 0, so Phi ≈ 0
    
    # Use erfc directly, but clip the argument to avoid overflow in exp
    # erfc is well-behaved for all real inputs in scipy
    
    phi_cdf = 0.5 * erfc(neg_z_over_sqrt2)
    
    # EI calculation
    ei = (f_best - mu) * phi_cdf + sigma_safe * phi_z
    
    # Add exploration bonus proportional to uncertainty
    # This makes the acquisition function more exploratory
    kappa = 0.5
    exploration_bonus = kappa * sigma_safe
    
    # Final score: higher means more attractive
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
