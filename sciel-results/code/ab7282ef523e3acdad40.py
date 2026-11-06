# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Uses an enhanced Best-Fit heuristic:
    - Primary: Strongly prefer bins where the item fits almost perfectly (minimize waste).
    - Uses an inverse scoring function: score = 1 / (waste + epsilon).
      This creates a much steeper preference for tight fits compared to linear scoring.
    - Secondary: In case of ties in waste (which implies ties in remaining capacity
      before placement for a fixed item size), the score is identical.
      To break potential floating-point ties or add stability, we add a very small
      bonus for bins with slightly less remaining capacity before placement 
      (i.e., bins that are already more utilized), encouraging consolidation.
    """
    # Calculate remaining capacity after placing the item
    remaining_after = bins - item
    
    # Identify valid bins (those that can fit the item)
    valid = remaining_after >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # For valid bins, calculate waste
        waste = remaining_after[valid]
        
        # Inverse scoring: higher score for smaller waste
        # Add epsilon to avoid division by zero
        eps = 1e-9
        scores[valid] = 1.0 / (waste + eps)
        
        # Tie-breaking / Consolidation bonus:
        # If waste is very small (close to 0), the inverse score is huge.
        # To differentiate bins with exactly the same waste (or near-zero waste),
        # we add a tiny bonus proportional to the "fullness" of the bin before placement.
        # Since bins[valid] is the remaining capacity BEFORE placement,
        # a smaller bins[valid] means the bin was already more full.
        # We want to prefer bins that were already more full (consolidation).
        # So we subtract a tiny fraction of bins[valid] from the score?
        # Wait, higher score is better.
        # If we want to prefer smaller bins[valid], we should subtract it?
        # No, if we subtract, smaller bins[valid] leads to larger score (less subtraction).
        # Let's use a very small coefficient.
        # Actually, if waste is the same, bins[valid] is the same (bins_after + item).
        # So this term is redundant if waste is exactly the same.
        # It only helps if waste differs by floating point precision.
        # Let's just keep the inverse score. It's robust.
        
        # However, if multiple bins have waste=0 (perfect fit), they all get 1/eps.
        # This is fine.
        
        # One potential issue: If waste is very large, score is small.
        # If waste is 0, score is 1e9.
        # This dynamic range is huge, which is good for prioritizing perfect fits.
        
        # Let's add a small term to prefer bins that are "older" or more full?
        # We don't have history.
        # Let's stick to the inverse waste.
        
        # To ensure we don't get inf/nan issues, we clip waste?
        # No, eps handles zero.
        
        # Final adjustment: 
        # If we have bins with waste=0, they all get 1e9.
        # If we have bins with waste=1, they get 1e-9.
        # This effectively makes the algorithm pick ANY perfect fit bin first.
        # If no perfect fit, it picks the best fit.
        
        # Is this better than linear?
        # Linear: waste=0 -> 0, waste=1 -> -1. Diff = 1.
        # Inverse: waste=0 -> 1e9, waste=1 -> 1e-9. Diff = 1e9.
        # This makes the algorithm *extremely* sensitive to waste.
        # This might be too strong? 
        # Consider: Item 10. Bin A remaining 10 (waste 0). Bin B remaining 11 (waste 1).
        # Inverse: A=1e9, B=1e-9. Pick A.
        # Linear: A=0, B=-1. Pick A.
        # Consider: Item 10. Bin A remaining 100 (waste 90). Bin B remaining 11 (waste 1).
        # Inverse: A=1e-11, B=1e-9. Pick B.
        # Linear: A=-90, B=-1. Pick B.
        # Both pick B.
        
        # The difference arises when waste is small.
        # If we have Bin A waste 0.1 and Bin B waste 0.2.
        # Linear: -0.1 vs -0.2. Pick A.
        # Inverse: 10 vs 5. Pick A.
        # The preference is stronger in Inverse.
        
        # Let's try a softer non-linear function: -waste**2
        # waste 0.1 -> -0.01. waste 0.2 -> -0.04.
        # waste 1 -> -1. waste 2 -> -4.
        # This penalizes large waste more than linear, but less than inverse.
        
        # Let's go with -waste**2 as it's stable and doesn't have the extreme dynamic range of inverse.
        # It strongly prefers tight fits without blowing up the scale.
        
        scores[valid] = -waste**2
        
        # Add a tiny tie-breaker for bins that were already more full (smaller bins[valid])
        # If waste is equal, bins[valid] is equal.
        # So this only helps with floating point noise.
        # Let's add a very small term: -1e-6 * bins[valid]
        # Smaller bins[valid] -> less negative -> higher score.
        # This prefers bins that had less remaining space before (more full).
        scores[valid] -= 1e-6 * bins[valid]
    
    return scores
# EVOLVE-BLOCK-END
