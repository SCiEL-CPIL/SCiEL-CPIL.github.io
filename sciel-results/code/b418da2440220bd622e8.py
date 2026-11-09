import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-6)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Candidate 1: Standard EI (Expected Improvement)
    # For minimization: E[f_best - f(x)]
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Candidate 2: LCB (Lower Confidence Bound) for minimization
    # We want to minimize mu - kappa * sigma.
    # Score should be high when mu - kappa * sigma is low.
    # score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # Increased kappa to 2.0 to encourage more exploration
    kappa = 2.0
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # Candidate 3: Thompson Sampling Proxy / Normalized Improvement Heuristic
    # This encourages sampling points with high potential relative to their uncertainty.
    # Score is (f_best - mu) / (sigma + epsilon)
    # This is essentially the z-score, but used as a raw score.
    # We scale it to be comparable to EI and LCB.
    # EI is roughly in the range [0, f_best - min(mu)]
    # LCB is roughly in the range [-max(mu) + kappa*max(sigma), ...]
    # The normalized score can be large. Let's scale it down.
    # A typical z-score for a good point might be 1-3.
    # Let's try multiplying by a small factor, or just use it as is and see.
    # Actually, (f_best - mu) / sigma is z.
    # If we want to encourage exploration, we might want to weight this more when sigma is small?
    # No, the directive says "maximizing the score (f_best - mu) / (sigma + epsilon)".
    # This is just the z-score.
    # Let's compute it.
    ts_scores = (f_best_safe - mu_safe) / sigma_safe
    
    # To combine these, we need them to be on similar scales.
    # EI is in objective units.
    # LCB is in objective units.
    # TS is dimensionless (ratio).
    # Let's try to normalize them or just use a weighted sum where TS has a small weight
    # or we scale TS by the typical scale of the objective.
    # However, the directive suggests using this as a primary or key component.
    # Let's try a hybrid:
    # 0.5 * EI + 0.3 * LCB + 0.2 * (TS * scale_factor)
    # What is a good scale_factor? Maybe the standard deviation of the observed values?
    # We don't have access to observed values, only f_best.
    # Let's assume the objective scale is around 1-10 for Hartmann 3D.
    # Let's try scaling TS by 1.0 first.
    
    # Alternative: Use the TS score to modulate the exploration/exploitation balance.
    # Or simply return a weighted sum.
    
    # Let's look at the best inspiration (score 1.9450).
    # It used: 0.6 * EI + 0.4 * LCB(kappa=2.0)
    # The current program (score 0.4493) used: EI + 0.5 * LCB(kappa=2.5)
    # The second inspiration (score 1.8129) used: 0.7 * EI + 0.3 * LCB(kappa=1.0)
    
    # The strategy to try is: "Use a Thompson Sampling proxy by approximating the expected minimum ... specifically by maximizing the score (f_best - mu) / (sigma + epsilon)"
    
    # Let's construct a score that blends EI and this TS proxy.
    # EI is a good exploitation measure.
    # TS proxy is a good exploration/relative improvement measure.
    
    # Let's try:
    # score = 0.5 * ei_scores + 0.5 * (ts_scores * 0.1)  # Scale down TS
    # Or maybe just use the TS score directly if it's more effective?
    # The directive says "maximizing the score ... which encourages sampling points with high potential relative to their uncertainty rather than using fixed linear combinations of mean and standard deviation."
    # This suggests replacing the linear combination with this ratio-based score.
    
    # Let's try a blend of EI and the TS proxy, but weight the TS proxy more heavily to follow the directive's spirit of "rather than using fixed linear combinations".
    
    # Let's try:
    # score = 0.3 * ei_scores + 0.7 * (ts_scores * 0.5)
    
    # Actually, let's look at the units.
    # EI: objective units
    # TS: dimensionless
    # If we just return ts_scores, the scale is different.
    # But the evaluator only cares about the *ranking* (argmax).
    # So the absolute scale doesn't matter as long as the relative ordering is good.
    # However, if we blend them, the scale matters.
    
    # Let's try a pure TS proxy first, as suggested by the directive's "rather than...".
    # But pure TS (z-score) might be too aggressive on exploration or not enough.
    # Let's try a mix.
    
    # Let's try:
    # score = 0.5 * ei_scores + 0.5 * ts_scores
    # This might not work well due to scale differences.
    
    # Let's try to normalize EI and TS to [0, 1] or similar?
    # No, that's expensive and might lose information.
    
    # Let's try the specific formula from the directive:
    # score = (f_best - mu) / (sigma + epsilon)
    # This is exactly ts_scores.
    
    # Let's try a combination that prioritizes this new direction but keeps EI for stability.
    # score = 0.4 * ei_scores + 0.6 * (ts_scores * 1.0)
    
    # Wait, if ts_scores is large (e.g. 10) and ei_scores is small (e.g. 0.1), the ts term dominates.
    # This might be good for exploration.
    
    # Let's try:
    final_scores = 0.3 * ei_scores + 0.7 * ts_scores
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return _sanitize(ei_scores)
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
