import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the "confidence" of the current best
    # If mu is close to f_best, the model is confident in the best region.
    # We want to balance exploitation (low mu) and exploration (high sigma).
    
    # Strategy: Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    
    # Adaptive Beta:
    # We estimate the "exploration potential" by looking at the ratio of uncertainty to the gap from the best.
    # gap = f_best - mu (positive if mu < f_best)
    # If the model predicts many points are much better than f_best, we should exploit those (low beta).
    # If the model is uncertain, we should explore (high beta).
    
    # Let's use a simple heuristic for beta:
    # beta = 1.0 + 1.0 * (std(sigma) / (std(mu) + 1e-9))
    # This increases beta if sigma varies a lot relative to mu.
    
    # Alternatively, a fixed beta is often robust.
    # Let's try a blend: 0.5 * EI + 0.5 * UCB? No, that's not a different family.
    
    # Let's try Pure UCB with a tuned beta.
    # Branin is 2D, range [-5,10]x[0,15].
    # Standard EI is score 1.0.
    
    # Let's try PI (Probability of Improvement) as it is a distinct family from EI and UCB.
    # PI = Phi((f_best - mu) / sigma)
    # PI is very greedy. It might get stuck.
    
    # Let's try **Expected Improvement with a Jitter** (Temperature).
    # This is effectively tuning EI.
    
    # Let's go back to **UCB**.
    # The key advantage of UCB over EI in early stages is that EI can be zero for points with high sigma but mu > f_best, whereas UCB will still give them a high score due to the sigma term.
    # This forces exploration of unexplored areas.
    
    beta = 1.5
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
