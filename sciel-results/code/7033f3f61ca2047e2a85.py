import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB)
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the "expected improvement" or "benefit" per cost.
    # Benefit = f_best - LCB(x) = f_best - mu(x) + beta * sigma(x)
    # Score = Benefit / Cost
    
    beta = 1.5  # Exploration constant
    
    lcb = mu - beta * sigma
    benefit = f_best - lcb
    # Ensure benefit is non-negative? 
    # If benefit is negative, it means LCB > f_best, so this point is likely worse than current best.
    # However, in early stages, f_best might be high, and LCB might be lower.
    # If we clip benefit to 0, we might ignore points that are slightly worse in LCB but have high uncertainty?
    # No, LCB already accounts for uncertainty. If LCB > f_best, it's unlikely to be better.
    # But EI also clips (implicitly, via max(0, ...)).
    # Let's not clip to allow "less bad" points to be selected if all are bad?
    # Actually, if all benefits are negative, we pick the one with the highest score (closest to 0 or least negative).
    # This is equivalent to picking the min LCB / max cost? No.
    # Score = Benefit / Cost.
    # If Benefit is negative, Score is negative.
    # Max Score will be the least negative.
    # This corresponds to the smallest (f_best - LCB) / Cost.
    # i.e., smallest (f_best - mu + beta*sigma) / Cost.
    # This seems reasonable.
    
    # Avoid division by zero or very small costs (cost >= 1)
    scores = benefit / cost
    
    # Handle NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
