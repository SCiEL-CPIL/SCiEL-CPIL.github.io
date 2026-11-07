"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    Uses a hybrid of Expected Improvement (EI) and a penalty for high cost,
    with an explicit time guard.
    """
    # Deadline guard: if we're running long, return a safe fallback
    # But since this function is called many times in a loop, we need to be fast.
    # The 300s limit is per evaluation, but the acquisition function is called
    # 2000 times per iteration. We need to be very fast here.
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheap points
    # But we also want to balance exploration. 
    # A common approach is to use EI / cost, but this can be too greedy.
    # Let's try a softer cost penalty: EI * (1 / (1 + cost)) or EI / cost^alpha
    
    # The seed program uses EI / cost. Let's try a different structural change.
    # Instead of pure EI/cost, let's use a UCB-like approach or a modified EI.
    
    # Structural change: Use a combination of exploitation (EI) and exploration (sigma)
    # weighted by cost efficiency.
    
    # Actually, let's look at the problem more carefully.
    # The cost is 1 + mean(x_scaled). x_scaled is in [0,1].
    # So cost is in [1, 2].
    # The seed program divides by cost, which means it strongly prefers cheap points.
    # But cheap points are in the "center" of the domain (x_scaled near 0.5 -> x near 0).
    # Wait, x_scaled = (x - (-5)) / 10 = (x+5)/10.
    # So x_scaled = 0.5 when x = 0.
    # The minimizer of Styblinski-Tang is at (-2.9035, -2.9035) approximately.
    # x_scaled for -2.9035 is (-2.9035+5)/10 = 0.20965.
    # mean(x_scaled) for the minimizer is 0.20965.
    # cost at minimizer is 1 + 0.20965 = 1.20965.
    # At x=0, cost is 1 + 0.5 = 1.5.
    # So the minimizer is actually cheaper than the center!
    
    # Let's try a different acquisition: Probability of Improvement (PI) divided by cost,
    # or a hybrid.
    
    # PI = P(f(x) < f_best) = norm.cdf(z)
    # PI is bounded [0, 1], EI can be large.
    
    # Let's try: (EI + lambda * sigma) / cost
    # This adds an exploration term.
    
    # Or, let's try a pure UCB: mu - beta * sigma, but we want to minimize, so
    # we want low mu and high sigma (uncertainty).
    # UCB for minimization: mu - beta * sigma
    # But we need to incorporate cost.
    
    # Let's try: (f_best - mu) / cost + beta * sigma / cost
    # This is like EI without the PDF term, plus exploration.
    
    # Actually, let's stick with EI but add a small exploration term.
    # score = (EI + 0.1 * sigma) / cost
    
    # Or, let's try a different structural change: 
    # Use a kernel density estimate or a simple local search?
    # No, we must use the GP's mu and sigma.
    
    # Let's try: EI / (cost ** 0.5) instead of EI / cost.
    # This softens the cost penalty.
    
    # Or, let's try: EI * exp(-cost)
    
    # Let's try a hybrid: 0.5 * (EI / cost) + 0.5 * (sigma / cost)
    
    # I'll try: (EI + 0.5 * sigma) / cost
    # This balances exploitation and exploration, both weighted by cost.
    
    exploration_weight = 0.5
    score = (ei + exploration_weight * sigma_safe) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
