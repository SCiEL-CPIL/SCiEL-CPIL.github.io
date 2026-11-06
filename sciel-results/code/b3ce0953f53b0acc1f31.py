"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We cache a lightweight "next-item" estimate derived from the occupancy
# distribution.  The function is stateless per call, but we can keep a small
# rolling history of item sizes to predict the next item's expected size.
# This is a structural change: instead of pure best-fit (min remaining after
# placing item), we score each bin by how well its remaining space matches
# the predicted next item size, reducing the chance of leaving a gap that
# future items cannot fill.

# Rolling history of recently placed item sizes (bounded)
_recent_items = []
_MAX_HISTORY = 200


def _predict_next_item_size(bins: np.ndarray, current_item: float) -> float:
    """
    Estimate the expected size of the next item using:
    1. The mean of recent items (if history is available).
    2. A blend with the current item size and the average remaining capacity
       of bins, to adapt to the current state of the packing.
    """
    global _recent_items
    _recent_items.append(current_item)
    if len(_recent_items) > _MAX_HISTORY:
        _recent_items = _recent_items[-_MAX_HISTORY:]
    
    if len(_recent_items) < 5:
        # Not enough history; fall back to a heuristic based on current state
        # Use the median of remaining capacities as a proxy for typical item size
        rem = bins - 0  # bins stores remaining capacity? 
        # Actually, from the seed, bins[i] is the remaining capacity.
        # Let's assume bins contains remaining capacity.
        # The item goes into a bin if bins[i] >= item.
        # After placement, bins[i] -= item.
        # So bins is remaining capacity.
        
        # Use mean of recent items if available, else current item
        if len(_recent_items) > 0:
            return float(np.mean(_recent_items))
        return current_item
    
    # Blend: 70% recent mean, 20% current item, 10% median remaining
    recent_mean = float(np.mean(_recent_items))
    rem_caps = bins[bins >= 0]  # all bins have remaining capacity
    if len(rem_caps) > 0:
        med_rem = float(np.median(rem_caps))
    else:
        med_rem = recent_mean
    
    predicted = 0.7 * recent_mean + 0.2 * current_item + 0.1 * med_rem
    return predicted


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Next-Item Awareness
    - Predict the expected size of the next item.
    - For each feasible bin, compute the remaining capacity after placing
      the current item.
    - Score the bin based on how well that remaining capacity matches the
      predicted next item size. A bin that leaves exactly the predicted next
      item size is ideal (score 0 distance). A bin that leaves too much or
      too little is penalized.
    - Also incorporate a small best-fit bias to prefer tighter fits for the
      current item, to avoid wasting large chunks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Identify feasible bins
    feasible_mask = bins >= item
    if not np.any(feasible_mask):
        # No feasible bin; return all -inf so that no bin is chosen
        # (the evaluator should handle this, but we return a valid array)
        scores = np.full_like(bins, -np.inf, dtype=np.float64)
        return scores
    
    # Predict next item size
    predicted_next = _predict_next_item_size(bins, item)
    
    # Compute remaining capacity after placing current item in each feasible bin
    remaining_after = bins[feasible_mask] - item
    
    # Score: minimize the absolute difference between remaining_after and predicted_next
    # This encourages bins that leave just enough space for the predicted next item.
    # Also add a small penalty for large remaining capacity (best-fit bias).
    # score = -(|remaining_after - predicted_next| + 0.1 * remaining_after)
    # Higher score is better, so we negate.
    
    diff = np.abs(remaining_after - predicted_next)
    best_fit_penalty = 0.1 * remaining_after  # small bias toward tighter fits
    bin_scores = -(diff + best_fit_penalty)
    
    # Build full scores array
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    scores[feasible_mask] = bin_scores
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found (which is already computed)
        return scores
    
    return scores
# EVOLVE-BLOCK-END
