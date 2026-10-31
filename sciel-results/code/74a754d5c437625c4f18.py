# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Hybrid Best/Worst Fit.
    - For large items (relative to average bin space), use Worst Fit to preserve 
      smaller bins for smaller future items? Or to ensure the large item goes 
      into a bin where it leaves the most flexibility? 
      Actually, putting large items in the bin with the most space (Worst Fit) 
      leaves the "tight" bins as they are, which might be better for small items?
      No, tight bins are bad for small items if they are too tight.
      
      Let's reconsider.
      If item is large, it eats up a lot of space.
      If we put it in a bin with lots of space (Worst Fit), we reduce that space.
      If we put it in a bin with little space (Best Fit), we reduce that space to near zero.
      
      Standard result: Best Fit is generally better.
      However, the seed's penalty might be too strong.
      
      Let's try a simple, robust **Best Fit** without the seed's extra penalty.
      score = -remaining
    """
    remaining = bins - item
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    feasible = remaining >= 0
    
    if np.any(feasible):
        # Standard Best Fit: minimize remaining space
        scores[feasible] = -remaining[feasible]
        
    return scores
# EVOLVE-BLOCK-END
