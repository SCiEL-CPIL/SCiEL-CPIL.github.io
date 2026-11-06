import numpy as np
import time

# Global state for adaptive heuristic
# We track the running mean of item sizes to adapt the target fill level.
# We also track a deadline guard to ensure we don't exceed time limits.
_start_time = time.time()
_running_mean = 50.0  # Initial guess for mean item size
_count = 0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time, _running_mean, _count
    
    # Deadline guard: If we are close to the limit, fall back to a fast static heuristic
    # The computation here is O(N), which is very fast, but we add the guard as requested.
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        # Fallback: Simple Best Fit (min remaining) to ensure we finish
        remaining = bins - item
        scores = np.full(len(bins), -np.inf)
        valid_mask = remaining >= 0
        scores[valid_mask] = -remaining[valid_mask]
        return scores

    # Update running mean of item sizes
    # Using a simple online mean update: new_mean = old_mean + (x - old_mean) / n
    if _count == 0:
        _running_mean = item
    else:
        _running_mean = _running_mean + (item - _running_mean) / (_count + 1)
    _count += 1
    
    # Calculate remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Adaptive Target Strategy:
    # We want to fill bins up to a certain level, but not too much.
    # The "target" remaining space is roughly 1.5 * mean_item_size.
    # If a bin has remaining space close to this target, it's a good candidate.
    # If a bin has much less remaining space, it's "too full" (risky for next items).
    # If a bin has much more remaining space, it's "too empty" (wasteful, but safe).
    # 
    # Score function: 
    # We prefer bins where the remaining space is close to the target.
    # However, we must prioritize feasibility. 
    # A common heuristic is to maximize the remaining space subject to a cap, 
    # or minimize the absolute difference from a target.
    # 
    # Let's try: Score = -|remaining - target|
    # But this might pick a bin with 0 remaining if target is 0.
    # We know target is > 0.
    # Actually, Best Fit is equivalent to minimizing remaining (maximizing fill).
    # Worst Fit is maximizing remaining.
    # 
    # Let's try a "Target Fill" approach:
    # Target remaining = alpha * mean_item_size
    # If remaining < target: We are filling a bin that is already quite full. 
    # If remaining > target: We are using a new bin or a very empty one.
    # 
    # A better adaptive strategy often seen in literature is to use a "decay" factor 
    # or to simulate the next item. 
    # 
    # Let's try a simple adaptive threshold:
    # If the item is small relative to the mean, be more aggressive (Best Fit).
    # If the item is large relative to the mean, be more conservative (Worst Fit).
    
    # Ratio of current item to mean
    ratio = item / _running_mean if _running_mean > 0 else 1.0
    
    # If ratio is high (large item), prefer bins with more remaining space (Worst Fit)
    # to avoid stranding the item in a tight space that might have been better left for smaller items.
    # If ratio is low (small item), prefer bins with less remaining space (Best Fit)
    # to consolidate small items.
    
    # Interpolate between Best Fit (score = -remaining) and Worst Fit (score = remaining)
    # weight = 0.5 + 0.5 * (ratio - 1) ? 
    # If ratio = 1, weight = 0.5 (balanced)
    # If ratio > 1 (large), weight > 0.5 (more Worst Fit)
    # If ratio < 1 (small), weight < 0.5 (more Best Fit)
    
    # Let's define a score that blends the two.
    # Best Fit score: -remaining (higher is better, so more negative is worse? No, we want max score. 
    # In the current code, scores[valid_mask] = -remaining[valid_mask]. 
    # So if remaining is 10, score is -10. If remaining is 1, score is -1. 
    # Max score is -1 (least remaining). This is Best Fit.
    
    # Worst Fit score: remaining (higher is better). 
    # If remaining is 10, score is 10. If remaining is 1, score is 1.
    # Max score is 10 (most remaining). This is Worst Fit.
    
    # We need a unified scale. Let's normalize.
    # Or simply:
    # If ratio > 1: Use Worst Fit (score = remaining)
    # If ratio < 1: Use Best Fit (score = -remaining)
    # This is a hard switch. Let's smooth it.
    
    # Let's use a power function.
    # Score = -remaining^p
    # If p=1, Best Fit.
    # If p=0.5, softer Best Fit.
    # If p=-1, Worst Fit (score = -1/remaining, max when remaining is max? No. -1/10 = -0.1, -1/1 = -1. Max is -0.1. So it picks max remaining. Yes.)
    
    # Let's interpolate p between 1.0 and -1.0 based on ratio.
    # If ratio = 1, p = 0? Score = -remaining^0 = -1. All valid bins have same score. Tie-breaker?
    # This is risky.
    
    # Alternative:
    # Score = -remaining + beta * (remaining / max_remaining)
    # This is getting complicated.
    
    # Let's stick to a simpler, robust adaptive heuristic:
    # "Harmonic" or "Geometric" mean of Best and Worst?
    
    # Let's try: Score = -remaining / (1 + ratio)
    # If ratio is large, denominator is large, score is close to 0 (small negative). 
    # All bins have small negative scores. Differences are small.
    # This doesn't seem to distinguish well.
    
    # Let's go back to the idea of Target Fill.
    # Target remaining = 1.5 * _running_mean
    # We want remaining to be close to Target.
    # Score = -abs(remaining - target)
    # This is a valid heuristic. It aims to keep bins at a consistent fill level.
    
    target = 1.5 * _running_mean
    
    scores = np.full(len(bins), -np.inf)
    
    if np.any(valid_mask):
        valid_remaining = remaining[valid_mask]
        # Calculate distance from target
        dist = np.abs(valid_remaining - target)
        # Score is negative distance
        scores[valid_mask] = -dist
        
    return scores
