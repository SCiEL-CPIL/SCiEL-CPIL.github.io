"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for tracking item statistics
_item_history = []
_MEDIAN_WINDOW = 100  # Keep last 100 items for median estimation
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # Safety margin in seconds


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Harmonic-mean-based scoring that models bin packing as a
    resource conservation task. We score bins based on the harmonic mean
    of their remaining capacity and the current item size, which
    prioritizes placements that minimize the relative waste of the bin's
    remaining potential compared to the item's demand.
    
    The harmonic mean H(a, b) = 2ab/(a+b) penalizes imbalance between
    remaining space and item size more strongly than arithmetic mean.
    This creates a nonlinear preference for placements where the remaining
    space is well-matched to the item, reducing fragmentation.
    
    We combine this with a best-fit component and a future-item fit proxy
    to balance tight packing with preserving capacity for future items.
    """
    global _item_history
    
    # Check deadline - if we're running out of time, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > 120.0 - _DEADLINE_MARGIN:
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
            scores[valid] += 1e-10 * bins[valid]  # tiebreaker
        return scores
    
    # Update item history
    _item_history.append(item)
    if len(_item_history) > _MEDIAN_WINDOW:
        _item_history = _item_history[-_MEDIAN_WINDOW:]
    
    # Estimate median item size from history
    if len(_item_history) >= 1:
        median_item = float(np.median(_item_history))
    else:
        median_item = item
    
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        waste = remaining[valid]
        
        # Component 1: Harmonic mean of remaining capacity and item size
        # H(waste, item) = 2 * waste * item / (waste + item)
        # Higher harmonic mean means better balance between remaining space and item
        # We want to MAXIMIZE this, so use it directly as positive score
        denom = waste + item
        # Avoid division by zero
        safe_denom = np.where(denom > 1e-12, denom, 1.0)
        harmonic_mean = 2.0 * waste * item / safe_denom
        
        # Component 2: Best-fit (minimize waste)
        # Use -waste so higher score = less waste
        best_fit_score = -waste
        
        # Component 3: Next-Item Fit proxy
        # Score based on how well the remaining space matches typical future items
        # We want remaining_after to be close to median_item
        if median_item > 1e-12:
            normalized_diff = (waste - median_item) / median_item
            next_item_score = -normalized_diff ** 2
        else:
            next_item_score = np.zeros_like(waste)
        
        # Component 4: Penalty for leaving very small fragments
        # If waste is much smaller than the item, we're creating a tiny fragment
        # that's hard to use. Apply a smooth penalty.
        fragment_penalty = np.where(waste < item * 0.1, -(item * 0.1 - waste) / (item * 0.1 + 1e-12), 0.0)
        
        # Combine all components with tuned weights
        # Harmonic mean is the primary signal (nonlinear resource conservation)
        # Best-fit provides a strong secondary signal
        # Next-item fit and fragment penalty are tertiary adjustments
        scores[valid] = (
            1.0 * harmonic_mean +
            0.3 * best_fit_score +
            0.1 * next_item_score +
            0.2 * fragment_penalty
        )
        
        # Tiny tiebreaker for numerical stability
        scores[valid] += 1e-10 * bins[valid]
    
    return scores


# EVOLVE-BLOCK-END
