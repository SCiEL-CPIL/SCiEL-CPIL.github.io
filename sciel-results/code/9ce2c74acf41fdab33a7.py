import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        rem_valid = remaining[valid]
        # Harmonic mean of remaining capacity and item size
        # H(a, b) = 2*a*b / (a + b)
        # We want to maximize the score, so we use negative harmonic mean
        # But harmonic mean is lower when one value is much smaller than the other
        # So -harmonic_mean will be higher (less negative) when the values are balanced
        
        # Calculate harmonic mean of rem_valid and item
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            h_mean = 2 * rem_valid * item / (rem_valid + item)
        
        # Where rem_valid + item is 0 (both 0), harmonic mean is 0
        h_mean = np.where((rem_valid + item) == 0, 0, h_mean)
        
        # Score is negative of harmonic mean
        scores[valid] = -h_mean
        
        # Additional penalty for very tight fits (fragmentation risk)
        very_tight = rem_valid < 5
        if np.any(very_tight):
            idx = np.where(valid)[0][very_tight]
            scores[idx] -= 0.1 * (5 - rem_valid[very_tight])
            
        # Bonus for moderate fits to encourage balanced usage
        moderate = (rem_valid >= 5) & (rem_valid <= 20)
        if np.any(moderate):
            idx = np.where(valid)[0][moderate]
            scores[idx] += 0.5
    return scores
