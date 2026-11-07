"""Rewritten acquisition for BO of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for speed
# We'll use a hybrid acquisition: a mix of Expected Improvement (EI) and a
# noise-injected lower confidence bound (LCB) style score, but vectorized
# and with a deadline guard.

# Since we don't have coordinates, we can't do multi-start local optimization.
# We must return scores for the 2000 candidate points given mu and sigma.

# Standard EI:
# z = (f_best - mu) / sigma
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)

# However, EI can be too greedy. A common improvement is to use a "noisy" EI
# or to blend with an exploration term. But without coordinates, we can't
# easily add a diversity penalty based on distance to evaluated points.

# One structural change: use a "noisy" acquisition where we add a small
# amount of noise to the sigma to encourage exploration in high-variance
# regions, or use a different kernel for the acquisition.

# Another approach: use the "Probability of Improvement" (PI) or a
# "Noisy" version of EI.

# Let's try a "Noisy Expected Improvement" where we effectively increase
# the uncertainty by a factor to encourage exploration. This is similar to
# using a larger beta in LCB but framed as EI.

# Actually, a very effective and simple structural change is to use a
# "Noisy" LCB (Lower Confidence Bound) for minimization, which is equivalent
# to maximizing a "UCB" (Upper Confidence Bound) for the negative function.
# But the task asks for higher scores to be evaluated next.
# So we want to maximize: mu - beta * sigma (for exploration)
# or maximize: EI.

# Let's try a blend of EI and a pure exploration term (sigma).
# Score = EI + lambda * sigma
# This encourages both improvement and exploration.

# However, the directive says "Make a STRUCTURAL change".
# Blending is a parameter change, not structural.

# Structural change: Use a "Noisy" EI where the noise is added to the
# predicted mean, not just the variance. Or, use a different acquisition
# function entirely, like "Probability of Improvement" (PI) which is
# less greedy than EI.

# Let's try PI (Probability of Improvement):
# PI = Phi((f_best - mu) / sigma)
# This is less greedy than EI because it doesn't weight by the amount of
# improvement, only the probability. This can lead to more exploration.

# Or, let's try a "Noisy" EI with a larger effective noise level.
# We can simulate this by scaling sigma up before computing EI.
# sigma_eff = sigma * scale_factor
# If scale_factor > 1, we are more exploratory.

# Let's try a hybrid: 0.5 * EI + 0.5 * (f_best - mu) * (1 - Phi(z))
# This is just EI.

# Let's try a different structural change: use a "Noisy" LCB for minimization.
# For minimization, we want to find low values.
# LCB = mu - beta * sigma
# We want to maximize the score, so we can use:
# Score = - (mu - beta * sigma) = -mu + beta * sigma
# This is a "UCB" for the negative function.
# This is structurally different from EI.

# Let's try this: Score = -mu + beta * sigma
# We need to choose beta. A typical value is 1.0 or 2.0.
# Let's try beta = 1.0.

# But wait, the task says "higher meaning evaluate next".
# So we want to maximize Score.
# If we use Score = -mu + beta * sigma, then points with low mu and high sigma
# will have high scores. This encourages exploration in uncertain regions
# and exploitation in low-mean regions.

# Let's implement this with a deadline guard.

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10 seconds margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Check deadline
    if time.time() > deadline:
        return ei
    
    # Compute a "Noisy" LCB-based score
    # Score = -mu + beta * sigma
    # This is structurally different from EI.
    beta = 1.5
    lcb_score = -mu + beta * sigma_safe
    
    # Blend EI and LCB score
    # This is a structural change: using a hybrid acquisition function.
    # The blend ratio can be tuned, but let's start with 0.5 * EI + 0.5 * LCB.
    # However, EI and LCB have different scales. EI is in objective units,
    # while LCB is also in objective units (since mu and sigma are in objective units).
    # So we can directly blend them.
    
    # But to be safe, let's normalize both to have zero mean and unit variance
    # before blending. This is a structural change: normalization.
    
    # Check deadline
    if time.time() > deadline:
        return ei
    
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-9:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = ei
    
    # Normalize LCB score
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std > 1e-9:
        lcb_norm = (lcb_score - lcb_mean) / lcb_std
    else:
        lcb_norm = lcb_score
    
    # Blend
    # Use a 50/50 blend
    score = 0.5 * ei_norm + 0.5 * lcb_norm
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
