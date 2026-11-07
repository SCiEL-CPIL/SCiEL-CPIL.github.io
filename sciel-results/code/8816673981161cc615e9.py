import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time
_start_time = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        # Return a safe, valid result
        return np.ones(len(mu)) * 1e-9

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Avoid division by zero in cost and sigma
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # Standard EI: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) for minimization
    # LCB = -mu + kappa * sigma
    # We want to minimize, so we look for low mu and high uncertainty.
    # Higher kappa encourages exploration.
    # Using a moderate kappa to balance exploration and exploitation.
    kappa = 2.0
    lcb = -mu + kappa * sigma_safe
    
    # 3. Combine EI and LCB
    # The current best program (score 1.3193) uses: scores = (ei + lcb) / cost
    # Inspiration 2 (score 1.0943) uses: scores = max(ei, ucb) / cost
    # Inspiration 1 (score 1.1056) uses normalized weighted sum.
    
    # The key insight from the "cell findings" is that PI (Probability of Improvement) 
    # weighted by inverse cost removed the exploration term of EI, focusing on 
    # high-confidence exploitation, which led to a jump from 1 to 1.803.
    # This suggests that for this specific problem (Griewank 2D, smooth, bounded),
    # over-exploration (high sigma term) might be harmful, or at least less effective
    # than pure exploitation or a very conservative blend.
    
    # However, pure EI (Inspiration 3, score 1.0000) was worse than the current hybrid.
    # The current hybrid (EI + LCB) is better than pure EI.
    
    # Let's look at the structure of the current program:
    # scores = (ei + lcb) / cost
    # ei ~ O(f_best)
    # lcb ~ O(f_best) + O(sigma)
    
    # A structural change: Instead of a linear combination, let's try a 
    # "Gaussian Process-UCB" style but with a cost-aware twist that favors 
    # exploitation when the GP is confident (low sigma) and exploration when 
    # uncertain, but dampened.
    
    # Alternative structural idea: Use the "Probability of Improvement" (PI) 
    # as a gate or weight for the EI, or simply use PI/cost as a baseline 
    # and add a small exploration term.
    
    # PI = Phi((f_best - mu) / sigma)
    # If we use PI / cost, we are rewarding points that are *likely* to be better, 
    # regardless of *how much* better. This is pure exploitation.
    
    # Let's try a blend that is more exploitation-heavy than the current one.
    # Current: ei + lcb = ei - mu + kappa*sigma
    # Since ei <= f_best - mu, the term -mu is partially canceled by ei.
    # Specifically, ei = (f_best - mu)Phi(z) + sigma phi(z).
    # So ei + lcb = (f_best - mu)Phi(z) + sigma phi(z) - mu + kappa*sigma
    #             = f_best*Phi(z) - mu(Phi(z)+1) + sigma(phi(z)+kappa)
    
    # Let's try a different combination: 
    # score = ei / cost + lambda * (sigma / cost) * indicator(low_confidence)
    
    # Actually, let's look at the "strongest ideas" from inspirations.
    # Inspiration 1 had a normalized weighted sum. 
    # Inspiration 2 had max(ei, lcb).
    
    # The current program is ei + lcb.
    # Let's try a multiplicative or gated approach.
    # If mu is very low (close to optimum), we want to exploit.
    # If mu is high but sigma is high, we might explore.
    
    # Let's try the "UCB with cost" directly, but with a lower kappa, 
    # and add a small EI term to break ties or boost exploitation.
    
    # Proposed structural change: 
    # Use a "Cost-Adjusted UCB" where the exploration term is scaled by 
    # the uncertainty relative to the best value.
    
    # Or, simpler: The finding says "PI weighted by inverse cost... effective".
    # Let's try: scores = (pi + alpha * ei) / cost
    # where pi = Phi(z)
    
    # Let's test this hypothesis.
    # If f_best is small, pi is the main driver.
    
    # Let's compute PI
    pi = norm.cdf(z)
    
    # Let's combine PI and EI.
    # PI is in [0, 1]. EI is in [0, f_best].
    # They are not on the same scale.
    
    # Let's try: scores = (pi * f_best + ei) / cost
    # This scales PI to objective units.
    
    # Or, let's stick to the current program's strength (EI+LCB) but 
    # modify the LCB to be less aggressive.
    
    # Current: lcb = -mu + 2.0 * sigma
    # Let's try: lcb = -mu + 1.0 * sigma
    
    # And let's try the max operation from Inspiration 2, but with the 
    # refined kappa.
    
    # Let's try: scores = np.maximum(ei, -mu + 1.5 * sigma_safe) / cost_safe
    
    # This is a structural change from sum to max.
    # Max ensures that if EI is high, we exploit. If LCB is high (due to sigma), we explore.
    # It prevents the "double counting" of uncertainty that might happen in the sum.
    
    # Let's go with this:
    kappa = 1.5
    lcb = -mu + kappa * sigma_safe
    
    scores = np.maximum(ei, lcb) / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
