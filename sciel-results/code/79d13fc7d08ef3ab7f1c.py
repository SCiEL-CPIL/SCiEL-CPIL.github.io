import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Best-Fit with Smooth Fragment Penalty.
    Base score is -remaining (Best-Fit).
    A smooth penalty is added for bins that would leave a very small remaining capacity,
    to avoid creating unusable fragments.
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        
        # Smooth fragment penalty
        # We penalize small remainders.
        # penalty = beta * exp(-gamma * rem)
        # This is high when rem is small, low when rem is large.
        beta = 5.0
        gamma = 0.5
        
        # Calculate penalty
        penalty = beta * np.exp(-gamma * rem)
        
        # Base best-fit score: -rem
        # Total score: -rem - penalty
        # Note: We want to MINIMIZE the score? No, the contract says "item goes to the highest-scoring bin".
        # So higher score is better.
        # Best-fit means we want SMALL remaining.
        # So score should be HIGH when remaining is SMALL.
        # score = -remaining means:
        #   rem=1 -> score=-1
        #   rem=10 -> score=-10
        # So -1 > -10, so we pick rem=1. Correct.
        
        # Now add penalty for small rem.
        # If rem is small, we want the score to be LOWER (worse) to avoid it?
        # Wait. If we avoid small fragments, we prefer a bin with slightly LARGER remaining.
        # So if rem is small, we want score to be lower than if it were slightly larger.
        # My formula: score = -rem - penalty.
        # If rem=0.1, penalty=5*exp(-0.05)≈4.75. score = -0.1 - 4.75 = -4.85.
        # If rem=1.0, penalty=5*exp(-0.5)≈3.03. score = -1.0 - 3.03 = -4.03.
        # -4.03 > -4.85. So we prefer rem=1.0 over rem=0.1.
        # This works. It pushes us away from tiny fragments.
        
        scores[valid] = -rem - penalty
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: pure best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
