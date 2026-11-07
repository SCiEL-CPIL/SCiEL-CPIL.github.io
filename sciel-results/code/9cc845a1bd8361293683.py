"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a One-Step Lookahead (Optimistic) strategy.
    
    Instead of a static combination of EI and UCB, this approach approximates
    the expected objective value after evaluating each candidate, assuming the
    GP is updated with that evaluation. This naturally balances exploitation
    (low mu) and exploration (high sigma) because evaluating a high-sigma point
    significantly reduces the posterior variance in that region, potentially
    revealing a much lower minimum in subsequent steps.
    
    Approximation:
    1. For each candidate x_i, we estimate the new mean mu_new and std sigma_new
       at x_i assuming we evaluate x_i and observe a value y.
    2. We use a "virtual" observation y* = mu_i - k * sigma_i (an optimistic 
       draw) to compute the posterior update at x_i.
    3. The score is based on the expected minimum value in a small neighborhood 
       around x_i after this update, or simply the updated mean at x_i.
       
    A simpler, robust proxy for lookahead in minimization is:
    Score = - (mu_i + lambda * sigma_i^2 / (mu_i + eps)) 
    But a more direct lookahead approximation is to score candidates by the 
    expected value of the *next* best point, which is hard.
    
    Alternative Materially Different Strategy: **Knowledge Gradient (KG) Approximation**
    The Knowledge Gradient measures the expected improvement in the best-found 
    value after taking an action. For minimization, it is the expected reduction 
    in f_best.
    
    Approximation of KG:
    KG_i ≈ E[ max(0, f_best - y_new) * I(y_new < f_best) + ... ]
    
    Actually, a very effective and computationally cheap alternative to EI/UCB 
    is the **Probability of Improvement (PI) weighted by the magnitude of 
    improvement**, but scaled by the *relative* uncertainty compared to the 
    current best improvement.
    
    Let's try a **Dynamic UCB** where the beta parameter is not fixed, but 
    adapts to the *scale* of the problem and the *progress* made.
    
    Current program uses EI + 0.3 * PI * sigma.
    Let's try a pure **Optimistic Value** approach:
    Score = -mu + beta(t) * sigma
    
    Where beta(t) is adapted based on the ratio of the current best improvement 
    to the standard deviation. If f_best is close to 0 (the minimum), we want 
    very low beta. If f_best is large, we want higher beta.
    
    However, the most materially different approach that often outperforms EI 
    in low-dimensional problems like Beale (smooth, single minimum) is 
    **Expected Hypervolume Improvement (EHVI)** or simply a **tuned UCB** with 
    a logarithmic time decay, but we don't have time index.
    
    Let's use **f_best-scaled UCB**:
    beta = c / (1 + f_best)
    This ensures aggressive exploration when f_best is large (early) and 
    exploitation when f_best is small (late).
    
    Wait, the current program already tries to adapt alpha but uses EI as base.
    Let's switch the base to UCB and make the adaptation explicit and stronger.
    
    Strategy: **Adaptive UCB with f_best normalization**
    score = -mu + beta * sigma
    beta = 2.0 / (1.0 + f_best)
    
    When f_best = 100, beta = 2/101 ≈ 0.02 (very exploitative? No, that's wrong. 
    If f_best is large, the function values are large, so sigma is likely large 
    in absolute terms. We need to explore. So beta should be HIGH when f_best 
    is large? 
    Actually, in BO, if f_best is far from the optimum, we are in a "search" 
    phase. We want to cover space. High sigma regions are good. 
    If f_best is close to optimum, we are in a "polish" phase. We want to 
    exploit low mu regions.
    
    So:
    High f_best -> High exploration -> High beta
    Low f_best -> Low exploration -> Low beta
    
    Let's use: beta = 1.5 * (1 + 1/f_best) ? No, if f_best is small, 1/f_best 
    is huge. We want beta to go DOWN as f_best goes DOWN.
    
    Let's use: beta = 1.5 / (1 + 1/f_best) ? 
    If f_best = 100, beta = 1.5 / 1.01 ≈ 1.48
    If f_best = 1, beta = 1.5 / 2 = 0.75
    If f_best = 0.1, beta = 1.5 / 11 ≈ 0.136
    
    This seems reasonable.
    
    Let's implement this Adaptive UCB.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta
    # Avoid division by zero if f_best is 0
    f_best_safe = max(f_best, 1e-6)
    
    # As f_best decreases, beta decreases.
    # We want to explore strongly when we are far from the minimum.
    # The Beale function minimum is 0.
    # Initial f_best might be ~100.
    # beta = 1.5 / (1 + 1/f_best_safe)
    # Let's verify the direction.
    # f_best=100 -> beta=1.5/1.01=1.48
    # f_best=1 -> beta=1.5/2=0.75
    # f_best=0.01 -> beta=1.5/101=0.0148
    # This looks correct.
    
    beta = 1.5 / (1.0 + 1.0 / f_best_safe)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score is better.
    score = -mu + beta * sigma_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
