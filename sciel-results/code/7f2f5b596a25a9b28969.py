import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    rem = bins - item
    C = 100.0
    # Sharper bimodal score: strongly prefers residuals near 0 or C
    # (rem * (C - rem)) is max at C/2, 0 at 0 and C.
    # Squaring it makes the preference for 0/C even stronger relative to the middle.
    base = rem * (C - rem)
    score = -base**2
    # Add a tiny linear bias towards Best Fit (smaller rem) to break ties
    # and because Best Fit is generally slightly better than Worst Fit for
    # minimizing bin count in this distribution.
    score += -0.1 * rem
    score = np.where(rem >= 0, score, -1e9)
    return score
