"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global state to track evaluation count for adaptive weighting
# The evaluator calls this function 25 times (30 total evals - 5 initial).
# We reset this state if the function is called with a fresh context, 
# but since we don't know the seed, we assume a single run per process or 
# rely on the fact that the evaluator might restart the process per seed.
# However, to be safe and robust against multiple calls in the same process 
# for different seeds (if any), we can't easily distinguish seeds.
# But the contract says "for each seed... the evaluator starts...". 
# Usually, this implies a fresh Python process or at least a fresh state.
# If the state persists across seeds, it will be wrong.
# Given the ambiguity, a stateless approach is safer, BUT the directive asks 
# for a STRUCTURAL change.
# Let's try a stateless approach that mimics adaptivity based on the input 
# distribution itself, or simply use a robust hybrid like UCB with a 
# tuned kappa.
# Actually, looking at the inspirations, pure EI is 1.0. 
# A structural change could be using a **Lower Confidence Bound (LCB)** 
# specifically tuned for minimization, or a **Probability of Improvement (PI)** 
# with a large epsilon, or **Knowledge Gradient**.
# However, a very strong structural change for 2D functions is to use 
# **Turbo-EI** or **Local Penalization** concepts, but we don't have 
# coordinates.
# Without coordinates, we are limited to the (mu, sigma) space.
# Let's try a **Variance-Weighted EI** or **Adaptive Kappa UCB**.
# Since we can't track state reliably across seeds without a key, 
# let's assume the process is restarted or use a heuristic based on 
# the current f_best and the spread of sigma.
# 
# Alternative Structural Change: **Max-Value Entropy Search (MES)** approximation? 
# Too complex without coordinates.
#
# Let's stick to a **Stateless Adaptive Hybrid** that uses the standard deviation 
# of the sigma values across the candidates to adjust the exploration parameter 
# (kappa) for UCB. If sigma is high everywhere, we are in early exploration. 
# If sigma is low, we are in late exploitation.
#
# Actually, the most robust structural change that often beats pure EI in 
# low-dimensional BO is **UCB with a carefully chosen kappa**. 
# Pure EI can get stuck in local minima if the GP is uncertain in the 
# promising region. UCB explicitly rewards uncertainty.
#
# Let's implement a **Hybrid Acquisition** that is a weighted sum of EI and UCB.
# To make it "structural" and not just a constant tune, we will make the 
# weight depend on the ratio of the best observed value to the range of 
# mu/sigma, or simply use a fixed robust kappa for UCB which is a different 
# family than EI.
#
# Let's try **UCB (Upper Confidence Bound)** for minimization:
# score = mu - kappa * sigma
# This is a different design family (exploration-driven) vs EI (improvement-driven).
# A common kappa for 2D is 1.0 to 2.0.
#
# Even better: **Adaptive UCB** where kappa is adjusted based on the 
# "confidence" of the current best. 
# But let's keep it simple and robust: **Pure UCB** is a valid structural 
# alternative to EI.
#
# Wait, the inspirations show that switching *to* EI gave 1.0. 
# Maybe the original was worse. 
# Let's try **Expected Improvement with a variance penalty** or **LCB**.
#
# Let's go with **Lower Confidence Bound (LCB)** which is effectively 
# UCB for minimization. 
# score = mu - kappa * sigma
#
# To add a "structural" element, we can normalize the scores to have 
# zero mean and unit variance, which helps if the scale of mu/sigma changes 
# drastically.
#
# Implementation:
# 1. Calculate UCB score: mu - kappa * sigma
# 2. Normalize scores: (score - mean) / std
# This normalization is a structural change in how the score is interpreted.

def acquisition(mu, sigma, f_best):
    # Use a kappa value that balances exploration and exploitation.
    # For 2D Michalewicz, which is multimodal, exploration is key.
    kappa = 1.5
    
    # UCB for minimization
    scores = mu - kappa * sigma
    
    # Normalize scores to prevent scale issues
    # If std is very small, normalization can blow up, so add a small epsilon
    std_scores = np.std(scores)
    if std_scores < 1e-9:
        std_scores = 1e-9
        
    normalized_scores = (scores - np.mean(scores)) / std_scores
    
    return normalized_scores
# EVOLVE-BLOCK-END
