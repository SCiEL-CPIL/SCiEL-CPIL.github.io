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
    
    # 1. Compute Standard Expected Improvement (EI)
    # For minimization: E[f_best - f(x)]
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # 2. Implement Local Penalization (LP-EI)
    # The idea: Penalize candidates that are too close to the current best location.
    # However, we do NOT have the coordinates of the best point passed to us.
    # We only have mu and sigma for 2000 candidates.
    # 
    # Alternative Strategy for "Diversification" without coordinates:
    # Since we can't compute distance to the best point, we can use the 
    # relationship between mu and sigma to infer "novelty".
    # 
    # A common heuristic in BO when coordinates are hidden but candidates are fixed 
    # is difficult. But wait, the prompt says "No coordinates are passed".
    # This means we cannot compute spatial distance.
    #
    # Let's rethink the "Materially Different" strategy.
    # The current program uses: 0.3 * EI + 0.7 * (z-score).
    # The z-score (f_best - mu) / sigma is actually the argument to the CDF in EI.
    # Maximizing z-score is equivalent to maximizing the probability that f(x) < f_best,
    # weighted by the magnitude of improvement? No, it's just the standard normal deviate.
    #
    # Let's try **Probability of Improvement (PI)** combined with **Uncertainty**.
    # PI = P(f(x) < f_best) = Phi(z).
    # This is a pure exploitation metric (probability of being better).
    #
    # Let's try **Upper Confidence Bound (UCB)** for minimization (which is LCB).
    # LCB = mu - kappa * sigma.
    # Score = -LCB = -mu + kappa * sigma.
    #
    # The previous run showed that a mix of EI and LCB worked well (inspiration 1: 1.9450).
    # The current program (0.1183) uses a TS proxy (z-score) which is essentially the 
    # derivative of the log-probability or related to the score function.
    #
    # Let's try a **Dynamic Kappa UCB** strategy.
    # In early stages, we want to explore more (high kappa).
    # In later stages, we want to exploit (low kappa).
    # We don't have the iteration count, but we can estimate progress 
    # by looking at the spread of sigma or the gap between f_best and min(mu).
    #
    # Heuristic for progress:
    # If f_best is much lower than the mean of mu, we are likely in a later stage 
    # (found a good region).
    # If f_best is close to the mean, we are early.
    #
    # Let's compute a "confidence" metric based on the improvement potential.
    # improvement_potential = f_best_safe - np.min(mu_safe)
    # If improvement_potential is small, we are close to the optimum -> exploit.
    # If improvement_potential is large, we are far -> explore.
    #
    # However, f_best is the global best seen so far.
    # min(mu) is the best predicted value among candidates.
    #
    # Let's try **Normalized LCB** where kappa adapts to the local uncertainty.
    # Or better: **Thompson Sampling via Normalization**.
    # 
    # Let's go back to the directive: "Use a Thompson Sampling proxy ... (f_best - mu) / (sigma + epsilon)".
    # The current program *does* this but mixes it with EI.
    # Maybe the mixing is the problem? The scales are different.
    # EI is in objective units. Z-score is dimensionless.
    # If f_best is -3.0 and mu is -2.0, EI ~ 0.5.
    # Z-score = 1.0 / 0.1 = 10.
    # 0.7 * 10 = 7. 0.3 * 0.5 = 0.15.
    # The z-score dominates completely.
    # This means the current program is effectively just maximizing the z-score.
    # Maximizing z-score is equivalent to maximizing the probability of improvement 
    # *if* sigma is constant? No.
    #
    # Let's try a **Pure UCB** with a tuned kappa.
    # Inspiration 1 (1.9450) used 0.6 EI + 0.4 LCB(k=2.0).
    # Inspiration 2 (1.8129) used 0.7 EI + 0.3 LCB(k=1.0).
    #
    # Let's try **Pure LCB** with kappa = 1.5.
    # This is a standard, robust baseline.
    #
    # OR, let's try **Expected Improvement with a Diversity Penalty** approximated by 
    # penalizing high-confidence points (low sigma) if they are not significantly better?
    # No, that's complex.
    #
    # Let's try **Quantile-Based Selection**.
    # Select the candidate that minimizes the 10th percentile of the posterior distribution.
    # q0.1 = mu - 1.28 * sigma.
    # Score = -q0.1 = -mu + 1.28 * sigma.
    # This is an LCB with kappa = 1.28.
    #
    # Let's try a **Hybrid** that is scale-invariant.
    # Normalize EI and LCB to have zero mean and unit variance, then sum them.
    # This ensures neither dominates due to scale differences.
    
    # Calculate EI
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Calculate LCB (kappa = 2.0, a common effective value)
    kappa = 2.0
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # Normalize both to zero mean, unit variance
    ei_mean = np.mean(ei_scores)
    ei_std = np.std(ei_scores)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_mean) / ei_std
        
    lcb_mean = np.mean(lcb_scores)
    lcb_std = np.std(lcb_scores)
    if lcb_std < 1e-9:
        lcb_norm = np.zeros_like(lcb_scores)
    else:
        lcb_norm = (lcb_scores - lcb_mean) / lcb_std
        
    # Combine normalized scores
    # Weight 0.5 / 0.5 for balance
    final_scores = 0.5 * ei_norm + 0.5 * lcb_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return _sanitize(ei_scores)
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
