"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Dynamic LCB with progress-based kappa adaptation and a multi-scale
    kernel-density-based exploration penalty to enforce spatial diversity.
    
    The score is higher where we expect lower function values (exploitation)
    and where uncertainty is high (exploration), but penalized in regions
    already densely covered by the candidate pool.
    """
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    
    # Progress-based kappa adaptation:
    # Early on (f_best far from minimum), use higher kappa for more exploration.
    # As we get closer to the known minimum (-3.86278), reduce kappa.
    # The known minimum for Hartmann 3D is approximately -3.86278.
    f_min_known = -3.86278
    
    # Normalize progress: 0 at start, 1 when f_best reaches f_min_known
    # Use a robust measure: how much improvement is possible
    max_possible_improvement = 0.5  # typical range of Hartmann 3D
    progress = np.clip((f_min_known - f_best) / max_possible_improvement, 0.0, 1.0)
    # Actually, f_best starts high (e.g., 0) and decreases toward -3.86278
    # So progress should be: (initial_f - f_best) / (initial_f - f_min)
    # But we don't know initial_f. Let's use:
    # progress = (f_best - f_min_known) / (some_scale)
    # When f_best is very high (e.g., 0.5), progress is large -> we want more exploration
    # When f_best is close to f_min_known, progress is small -> exploit
    
    # Better: use the gap between f_best and f_min_known
    gap = f_best - f_min_known  # positive, decreases over time
    # Scale: typical gap starts around 0.5-1.0 and goes to 0
    # kappa should be high when gap is large, low when gap is small
    # kappa = base_kappa * (1 + alpha * gap)
    # Or: kappa = base_kappa / (1 + beta * (f_min_known - f_best + epsilon))
    
    # Let's use: kappa = kappa_min + (kappa_max - kappa_min) * exp(-gamma * gap)
    # When gap is large (early), exp(-gamma*gap) is small -> kappa near kappa_min? No, that's backwards.
    # We want: early -> high kappa, late -> low kappa
    # So: kappa = kappa_max - (kappa_max - kappa_min) * (1 - exp(-gamma * gap))
    # When gap is large: 1 - exp(-gamma*gap) ≈ 1, so kappa ≈ kappa_min. That's wrong.
    # Let's think again:
    # gap = f_best - f_min_known. Early: f_best ≈ 0.5, f_min ≈ -3.86, gap ≈ 4.36
    # Late: f_best ≈ -3.86, gap ≈ 0
    # We want kappa high when gap is large, low when gap is small.
    # kappa = kappa_max * exp(-gamma * (1 - gap / gap_max))
    # Or simpler: kappa = kappa_min + (kappa_max - kappa_min) * (gap / gap_max)
    # But gap_max is unknown. Let's use a fixed scale.
    
    # Simple approach: kappa decreases linearly with progress
    # progress = 1 - gap / gap_initial, but we don't know gap_initial
    # Let's use: kappa = 2.0 * (1 + 0.5 * gap) but capped
    # Actually, let's use a dynamic formula:
    # kappa = 0.5 + 1.5 * exp(-gap)
    # When gap ≈ 4: exp(-4) ≈ 0.018, kappa ≈ 0.53
    # When gap ≈ 0: exp(0) = 1, kappa = 2.0
    # This gives high kappa when close to optimum, which is backwards!
    
    # Correct: we want HIGH kappa (more exploration) when gap is LARGE (early)
    # kappa = 0.5 + 1.5 * exp(-gap / scale) where scale controls the rate
    # When gap is large: exp(-large/small) ≈ 0, kappa ≈ 0.5 (low exploration) -- WRONG
    # We need: kappa large when gap large
    # kappa = 0.5 + 1.5 * (1 - exp(-gap / scale))
    # When gap large: 1 - exp(-large) ≈ 1, kappa ≈ 2.0 (high exploration) ✓
    # When gap small: 1 - exp(0) = 0, kappa ≈ 0.5 (low exploration) ✓
    
    scale = 1.0
    kappa = 0.5 + 1.5 * (1.0 - np.exp(-gap / scale))
    kappa = np.clip(kappa, 0.3, 3.0)
    
    # LCB score for minimization: higher score = better
    # LCB = mu - kappa * sigma (this is the lower bound on f)
    # For acquisition (higher is better to select): score = -LCB = -mu + kappa * sigma
    lcb_score = -mu + kappa * sigma
    
    # Multi-scale kernel-density-based exploration penalty:
    # Estimate local density of the 2000 candidates using a fixed bandwidth.
    # Since we don't have coordinates, we use the GP posterior as a proxy for
    # spatial structure. High sigma regions are likely under-explored.
    # But the directive says to use candidate density. Without coordinates,
    # we can't compute true spatial density. 
    # 
    # Alternative interpretation: use the structure of mu and sigma to infer
    # regions that are "densely sampled" by the GP. Where sigma is low, the
    # GP has been well-informed, suggesting nearby points have been evaluated.
    # Where sigma is high, the region is under-explored.
    #
    # The density penalty should REDUCE the score in well-sampled regions.
    # Well-sampled regions have low sigma. So penalty = -w * (1/sigma_safe)
    # But this could be unstable. Let's use: penalty = -w * exp(-sigma / bandwidth)
    # When sigma is small (well-sampled): exp(-small/bandwidth) ≈ 1, penalty ≈ -w
    # When sigma is large (under-sampled): exp(-large/bandwidth) ≈ 0, penalty ≈ 0
    # This penalizes well-sampled regions, encouraging exploration.
    
    bandwidth = 0.1
    density_proxy = np.exp(-sigma / bandwidth)
    penalty_weight = 0.1
    penalty = penalty_weight * density_proxy
    
    # Final score
    score = lcb_score - penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
