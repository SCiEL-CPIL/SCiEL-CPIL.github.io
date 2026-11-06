"""Online bin packing priority function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


# Global state for adaptive next-item estimation
_state = {
    'remaining': [],  # list of remaining capacities
    'recent_items': [],  # recent item sizes (ring buffer)
    'window': 20  # window size for estimating next item
}


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred (item goes to highest-scoring bin).
    
    Strategy: Best-Fit with Next-Item Awareness.
    We estimate the expected size of the next item from recent history
    and adjust the best-fit score to prefer bins whose remaining space
    after placing the current item is close to the predicted next item.
    This reduces fragmentation by consolidating items of similar size.
    """
    # Update state with current bins
    _state['remaining'] = list(bins)
    _state['recent_items'].append(item)
    if len(_state['recent_items']) > _state['window']:
        _state['recent_items'].pop(0)
    
    # Estimate expected next item size
    # Use median of recent items as a robust estimator
    if len(_state['recent_items']) >= 2:
        next_item_est = np.median(_state['recent_items'])
    else:
        # Fallback: use current item as proxy
        next_item_est = item
    
    # Compute remaining capacity if item is placed
    rem = bins - item
    
    # Initialize scores
    scores = np.full(len(bins), -1e18)
    feasible = rem >= 0
    
    if np.any(feasible):
        feasible_rem = rem[feasible]
        
        # Base best-fit score: prefer smaller remaining space
        best_fit_score = -feasible_rem
        
        # Next-item awareness score: prefer remaining space close to next_item_est
        # We want to maximize how well remaining space matches next_item_est
        # Use negative absolute difference: -|rem - next_item_est|
        # But we also want to strongly prefer feasible bins
        next_item_score = -np.abs(feasible_rem - next_item_est)
        
        # Blend: primarily best-fit, with next-item awareness
        # The weight on next-item awareness should be moderate
        # If next_item_est is very different from typical remaining spaces, 
        # the next-item score may dominate incorrectly, so we normalize
        
        # Normalize both scores to comparable scales
        if len(feasible_rem) > 1:
            min_bf = best_fit_score.min()
            max_bf = best_fit_score.max()
            if max_bf > min_bf:
                norm_bf = (best_fit_score - min_bf) / (max_bf - min_bf)
            else:
                norm_bf = np.zeros_like(best_fit_score)
            
            min_ni = next_item_score.min()
            max_ni = next_item_score.max()
            if max_ni > min_ni:
                norm_ni = (next_item_score - min_ni) / (max_ni - min_ni)
            else:
                norm_ni = np.zeros_like(next_item_score)
            
            # Blend: 70% best-fit, 30% next-item awareness
            blended = 0.7 * norm_bf + 0.3 * norm_ni
        else:
            blended = np.array([1.0])
        
        scores[feasible] = blended
    
    return scores
# EVOLVE-BLOCK-END
