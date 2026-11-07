import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # The optimistic estimate is mu - beta * sigma.
    # We want to minimize this, so we maximize (beta * sigma - mu).
    # Cost-aware score: (beta * sigma - mu) / cost
    
    # Beta controls the exploration/exploitation trade-off.
    # A larger beta encourages exploration of high-uncertainty regions.
    # Given the small budget and multimodal nature, we use a moderate-to-high beta.
    beta = 3.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate the unnormalized score
    # Note: mu and sigma are in objective units.
    # If f_best is very low, and mu is high, (beta*sigma - mu) might be negative.
    # We should probably clip the score to be non-negative?
    # If beta*sigma < mu, the LCB is higher than mu, meaning we are not optimistic about improvement.
    # However, standard LCB just picks the min of (mu - beta*sigma).
    # Maximizing (beta*sigma - mu) is equivalent.
    
    raw_scores = beta * sigma_safe - mu
    
    # It's possible for raw_scores to be negative if mu is very large relative to beta*sigma.
    # In that case, the point is likely worse than current best even with exploration.
    # We can clip to 0 to avoid selecting such points if all are negative, 
    # or just let the normalization handle it. 
    # However, "higher = evaluate next". If all are negative, the least negative is best.
    # Clipping to 0 might be safer to prevent selecting points that are definitely worse?
    # Actually, if all points are worse, we still have to pick one.
    # Let's not clip, but ensure numerical stability.
    
    safe_cost = cost + 1e-9
    scores = raw_scores / safe_cost
    
    return scores
