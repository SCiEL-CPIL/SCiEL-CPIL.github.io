"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Enforce deadline with safety margin
    start_time = time.time()
    deadline = start_time + 295.0  # 300s limit, 5s safety margin

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zeros in sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Pre-compute CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (uncosted)
    EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Strategy: Expected Hypervolume Improvement (EHVI) approximation for single-objective
    # For minimization, we can transform to maximization by negating, or use the direct formula.
    # The EHVI for a single point x with posterior N(mu, sigma^2) relative to a reference point r
    # (here r = f_best) is proportional to:
    # EI + sigma * (1 - cdf(z)) * log(1 + sigma^2 / (f_best - mu + sigma^2)^2) ... wait, that's complex.
    
    # A simpler and robust proxy for EHVI in single-objective BO that captures both
    # exploitation (EI) and exploration (uncertainty weighted by potential improvement)
    # is the "Expected Improvement" itself, but we need to handle the cost.
    
    # The current program uses EI / cost^0.7 + exploration_bonus.
    # The exploration bonus in the current program is somewhat arbitrary.
    
    # Let's try a different formulation:
    # Score = (EI + kappa * sigma * (1 - cdf(z))) / cost^alpha
    # Where the term sigma * (1 - cdf(z)) represents the "upper confidence bound" style exploration
    # weighted by the probability of *not* being better, which acts as a penalty for high uncertainty
    # if it's likely to be bad, but rewards uncertainty if it might be good? No.
    
    # Actually, the standard UCB is mu + kappa * sigma.
    # For minimization, we want low mu and low sigma? No, low mu is good, high sigma is exploration.
    # We want to minimize f(x).
    # Improvement is f_best - f(x).
    # E[Improvement] = EI.
    
    # Let's look at the "Expected Improvement" again.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma.
    
    # A known issue with EI is that it can be too greedy.
    # The "Expected Hypervolume Improvement" for a single point is:
    # EHVI = EI + (sigma / sqrt(2*pi)) * exp(-z^2/2) * (something related to the reference point distance)
    
    # However, a simpler and effective alternative is to use a **Weighted Expected Improvement**
    # or a **Cost-Effective Expected Improvement** with a specific cost model.
    
    # Let's try a different cost-aware metric:
    # The cost is c = 1 + mean(x_scaled).
    # We want to maximize (f_best - f(x)) / c.
    # This is the "Expected Improvement per unit cost".
    # E[ (f_best - f(x)) / c ] is not exactly EI/c because c is deterministic given x, but EI is the expectation over f(x).
    # Since c is deterministic for a candidate x, we can compute:
    # Score = E[ max(0, f_best - f(x)) ] / c(x) = EI(x) / c(x)
    # This is what the current program approximates with EI / cost^0.7.
    # Why 0.7? Maybe to be less punitive.
    
    # Let's try **Expected Improvement** with a **Gaussian Process Lower Confidence Bound (GP-UCB)** style
    # but adapted for cost.
    
    # Actually, let's look at the "Inspiration 1" which scored 1.41.
    # It likely used a more sophisticated acquisition.
    
    # Let's try **Expected Improvement** with a **dynamic exploration parameter**.
    # Or better, let's use **Expected Hypervolume Improvement (EHVI)** properly.
    
    # For a single objective, EHVI is:
    # EHVI = EI + (sigma * (1 - Phi(z))) * log(1 + (sigma / (f_best - mu))^2) ... no, that's not right.
    
    # Let's use the **Expected Improvement** but with a **penalty for high cost** that is
    # more aggressive in the early stages and less aggressive later? No, we don't have time steps.
    
    # Let's try a **Hybrid** of EI and UCB.
    # UCB for minimization: mu - kappa * sigma.
    # We want to minimize mu - kappa * sigma.
    # So score = -(mu - kappa * sigma) / cost^alpha = (kappa * sigma - mu) / cost^alpha.
    # This combines exploitation (low mu) and exploration (high sigma).
    
    # Let's tune kappa. A common value is 1.0 or 2.0.
    # Let's try kappa = 1.0 and alpha = 0.5 (square root cost penalty).
    
    # Wait, EI already includes an implicit exploration term (sigma * phi(z)).
    # If we use UCB, we might explore too much if kappa is large.
    # If we use EI, we might exploit too much.
    
    # Let's try a **normalized** approach.
    
    # Calculate EI
    EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Calculate UCB (for minimization, lower is better, so we maximize -UCB)
    # UCB = mu - kappa * sigma
    # Score_UCB = -UCB = kappa * sigma - mu
    kappa = 1.5
    Score_UCB = kappa * sigma_safe - mu
    
    # Normalize both to [0, 1] roughly to combine them
    # But normalization depends on the batch.
    
    # Let's try a simple weighted sum of EI and UCB, both divided by cost.
    # Score = (w1 * EI + w2 * (kappa * sigma - mu)) / cost^alpha
    
    # Let's use w1 = 1.0, w2 = 0.5, alpha = 0.5.
    
    cost_penalty = np.power(cost, 0.5)
    
    score_ei = EI / cost_penalty
    score_ucb = Score_UCB / cost_penalty
    
    # Ensure scores are positive? EI is always >= 0. UCB can be negative.
    # If UCB is negative, it means mu > kappa * sigma, so it's likely bad.
    # We should clip UCB to 0? Or just let it be.
    
    # Let's blend them.
    score = 0.7 * score_ei + 0.3 * np.maximum(score_ucb, 0)
    
    # Safety: ensure all scores are finite
    if not np.all(np.isfinite(score)):
        score_fallback = EI / np.maximum(cost, eps)
        if np.all(np.isfinite(score_fallback)):
            score = score_fallback
        else:
            score = np.zeros_like(mu)
    
    # Check time budget
    if time.time() > deadline:
        pass
    
    return score
# EVOLVE-BLOCK-END
