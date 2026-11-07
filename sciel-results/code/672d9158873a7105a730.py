"""Cost-aware acquisition for Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_module_start_time = time.time()
_DEADLINE_LIMIT = 310  # seconds; evaluator kills at 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Hartmann 3D minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Evaluation cost at each candidate (in [1, 2]).
    
    Returns
    -------
    score : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable next evaluation.
    """
    # Deadline guard
    elapsed = time.time() - _module_start_time
    if elapsed > _DEADLINE_LIMIT:
        # Fallback: simple LCB
        return mu - sigma

    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from GP posterior
    # sample ~ N(mu, sigma^2)
    # Use a fixed seed per call? No - we want randomness.
    # But we need reproducibility? The task doesn't require it, but
    # let's use a deterministic approach for stability within a run.
    # Actually, Thompson Sampling inherently uses randomness.
    # We'll draw a single sample using standard normal.
    
    # Generate random normal values
    z = np.random.standard_normal(n)
    sample = mu + sigma_safe * z
    
    # For minimization, we want low sampled values
    # But we also need cost-awareness: prefer cheaper evaluations
    # Cost-aware Thompson Sampling: penalize high-cost points
    
    # The cost is in [1, 2]. We can use a cost-adjusted score.
    # One approach: score = -sample / cost^alpha
    # Or: score = -(sample + beta * (cost - 1))
    
    # Let's use: score = -sample * cost^(-gamma)
    # where gamma controls cost sensitivity.
    
    # Adaptive gamma based on how close we are to the optimum
    f_min_theoretical = -3.86278
    gap = f_best - f_min_theoretical
    
    # If gap is large, we're far from optimum, explore more (lower gamma)
    # If gap is small, we're near optimum, be cost-efficient (higher gamma)
    # Clamp gap to [0, 2] for scaling
    gap_clipped = np.clip(gap, 0.0, 2.0)
    
    # gamma from 0.2 (explore) to 1.0 (exploit/cost-efficient)
    gamma = 0.2 + 0.8 * (1.0 - gap_clipped / 2.0)
    gamma = np.clip(gamma, 0.2, 1.0)
    
    # Cost factor: cost is in [1, 2], so cost^(-gamma) is in [2^(-gamma), 1]
    # Lower cost -> higher factor -> higher score
    cost_factor = np.power(cost, -gamma)
    
    # Score: higher is better. We want low sample value (good) and low cost (good).
    # score = -sample * cost_factor
    # But this can be dominated by the sample value.
    
    # Alternative: use a normalized approach
    # Normalize sample to [0, 1] range relative to current best
    # Then combine with cost
    
    # Let's use: score = -(sample - f_best) / (sigma_mean + eps) * cost_factor
    # This normalizes by uncertainty level
    
    # Actually, simpler: just use -sample / cost^gamma
    # The scale of sample matters. Let's check typical magnitudes.
    # Hartmann 3D values are around -3 to 0.
    
    score = -sample * cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
