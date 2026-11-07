import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Mean Variance (MV) acquisition
    # MV = f_best - mu + kappa * sigma
    # This is a linear combination that balances exploitation and exploration more directly
    # than EI, which can be overly greedy or overly exploratory depending on the shape of the GP.
    # Also, MV is computationally cheaper and often more robust.
    # We can use a weighted sum of EI and MV, or just MV with a cost penalty.
    # Let's try a hybrid: Score = (EI + MV) / cost^alpha
    # Or simply: Score = (f_best - mu + kappa * sigma) / cost^alpha
    
    # Let's use a dynamic kappa for MV.
    # Early on, we want high exploration (high kappa).
    # Late on, we want exploitation (low kappa).
    # f_best starts high and drops.
    # Let's map f_best to kappa.
    # If f_best > 5, kappa = 2.0
    # If f_best < 0, kappa = 0.5
    # Linear interpolation: kappa = 0.5 + 1.5 * (1 - min(f_best, 5)/5)
    # This is getting complex. Let's stick to EI but try a different cost normalization.
    
    # The previous solution used alpha = 1.0 + 0.5 * exp(-max(f_best, 0)).
    # This resulted in alpha ~ 1.5 early, ~ 1.0 late.
    # This means the cost penalty is STRONGER early.
    # But early, we have little data, so sigma is high everywhere.
    # If we penalize cost heavily early, we might get stuck in a local minimum in the low-cost region.
    # We should probably penalize cost LESS early and MORE late.
    # So alpha should be LOWER early and HIGHER late.
    # Current: alpha = 1.0 + 0.5 * exp(-max(f_best, 0))
    # f_best=100 -> alpha=1.0
    # f_best=0 -> alpha=1.5
    # This is actually correct: lower penalty early, higher penalty late.
    
    # What if we use a different acquisition function entirely?
    # Lower Confidence Bound (LCB) for minimization:
    # LCB = mu - kappa * sigma
    # We want to minimize LCB, so we maximize -LCB = -mu + kappa * sigma.
    # Improvement = f_best - LCB = f_best - mu + kappa * sigma.
    # This is similar to EI but without the non-linearity of the normal CDF.
    # Let's try LCB with a dynamic kappa.
    
    # Dynamic kappa for LCB:
    # Early: high kappa (explore)
    # Late: low kappa (exploit)
    # f_best high -> high kappa
    # f_best low -> low kappa
    # kappa = 1.0 + 1.0 * exp(-max(f_best, 0))
    # f_best=100 -> kappa=1.0
    # f_best=0 -> kappa=2.0
    # Wait, I want high kappa early.
    # So if f_best is high, I want high kappa.
    # kappa = 1.0 + 1.0 * (1 - exp(-max(f_best, 0)))
    # f_best=100 -> kappa=2.0
    # f_best=0 -> kappa=1.0
    # This looks better.
    
    # Let's combine EI and LCB.
    # Score = (EI + LCB_improvement) / cost^alpha
    
    # LCB improvement
    kappa = 1.0 + 1.0 * (1.0 - np.exp(-np.maximum(f_best, 0.0)))
    lcb_improvement = f_best - mu + kappa * sigma_safe
    lcb_improvement = np.maximum(lcb_improvement, 0.0)
    
    # Average EI and LCB
    combined = 0.5 * ei + 0.5 * lcb_improvement
    
    # Cost penalty
    alpha = 1.0 + 0.5 * np.exp(-np.maximum(f_best, 0.0))
    
    scores = combined / np.power(cost_safe, alpha)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
