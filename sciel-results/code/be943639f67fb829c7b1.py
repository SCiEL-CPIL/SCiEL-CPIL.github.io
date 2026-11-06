import numpy as np
import time
from collections import deque

# Global state
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s kill
_recent_items = deque(maxlen=200)
_min_future_item = 1.0  # Initial estimate

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global START_TIME, DEADLINE, _recent_items, _min_future_item
    
    # Update recent items history
    _recent_items.append(item)
    
    # Estimate minimum future item size to identify "dead" gaps
    # We use a low percentile to be conservative; if a gap is smaller than this,
    # it's likely too small for most future items.
    if len(_recent_items) >= 20:
        # 10th percentile
        _min_future_item = np.percentile(list(_recent_items), 10)
    else:
        # If not enough history, use a fraction of the current item
        _min_future_item = item * 0.5 if item > 0 else 1.0

    # Check deadline
    if time.time() - START_TIME > DEADLINE:
        # Fallback to simple Best Fit if running out of time
        remaining = bins - item
        scores = np.full_like(bins, -np.inf, dtype=float)
        valid_mask = remaining >= 0
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    remaining = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining[valid_mask]
    
    # Base score: Best Fit (prefer smaller remaining space after placement)
    # Higher score is better. -remaining means smaller remaining -> higher score.
    base_scores = -valid_remaining
    
    # Identify bins where the remaining space would be "dead"
    # (too small to fit the estimated minimum future item)
    dead_mask = valid_remaining < _min_future_item
    
    # Apply a large penalty to dead gaps to avoid them if possible
    penalty = 100.0
    base_scores[dead_mask] -= penalty
    
    scores[valid_mask] = base_scores
    
    return scores
