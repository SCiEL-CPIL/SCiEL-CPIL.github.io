"""Low discrepancy on all arithmetic progressions - Greedy Worst-AP Repair with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Also returns the step size 'd' and start index 'i' of one such worst AP.
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns (max_abs, d, i).
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_d = 1
    best_i = 0
    
    for d in range(1, n + 1):
        # For a given d, iterate over all starting indices i in [0, d)
        # such that i < n.
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                best_d = d
                best_i = i
    return int(max_abs), best_d, best_i

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc, current_d, current_i = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Greedy Repair Loop
    # Strategy:
    # 1. Find the AP (d, i) with max discrepancy.
    # 2. For each element in that AP, simulate flipping it.
    # 3. Evaluate the new global discrepancy.
    # 4. Pick the flip that yields the lowest discrepancy.
    # 5. If improvement, accept. If no improvement, try random flips or stop.
    
    while time.time() < deadline:
        # Find current worst AP
        current_disc, current_d, current_i = _get_max_ap_discrepancy(signs)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs.copy()
        
        # Identify indices in the worst AP
        # AP: i, i+d, i+2d, ... < n
        ap_indices = list(range(current_i, n, current_d))
        
        if len(ap_indices) == 0:
            break
            
        best_flip_idx = -1
        best_flip_disc = current_disc
        best_flip_val = None
        
        # Evaluate flipping each index in the AP
        for idx in ap_indices:
            # Flip
            signs[idx] = -signs[idx]
            
            # Evaluate new discrepancy
            # Optimization: We can potentially prune if we know the max possible reduction,
            # but for n=256, full evaluation is ~1-2ms. 
            # Let's do full evaluation to be safe and simple.
            new_disc, _, _ = _get_max_ap_discrepancy(signs)
            
            if new_disc < best_flip_disc:
                best_flip_disc = new_disc
                best_flip_idx = idx
                best_flip_val = signs[idx]
                
            # Revert for next trial
            signs[idx] = -signs[idx]
        
        # Apply the best flip
        if best_flip_idx != -1 and best_flip_disc < current_disc:
            signs[best_flip_idx] = best_flip_val
            # current_disc will be updated in next loop iteration's check
            # But we can update local variable for logic if needed
        else:
            # No improvement found by flipping within the worst AP.
            # Try a few random flips to escape local minima?
            # Or just break if we've stuck?
            # Let's try a random flip with a small probability to explore, 
            # but if we are stuck for too long, we might want to restart or accept.
            # Given the time constraint, if greedy fails, random walk is risky.
            # Let's try a random flip once. If it improves, take it. Else, break?
            # Actually, if greedy on the worst AP doesn't improve, the solution might be locally optimal 
            # with respect to single flips in that AP. 
            # Let's try a random index flip.
            
            improved = False
            for _ in range(5): # Try a few random flips
                idx = rng.integers(0, n)
                signs[idx] = -signs[idx]
                new_disc, _, _ = _get_max_ap_discrepancy(signs)
                if new_disc < current_disc:
                    # Keep this flip
                    current_disc = new_disc
                    improved = True
                    break
                else:
                    # Revert
                    signs[idx] = -signs[idx]
            
            if not improved:
                # Stuck. Break to save time or return best.
                # Since we can't easily backtrack, we just stop.
                break
        
    # Final check to ensure best_signs is the absolute best found
    # (In case the last accepted flip was better than best_signs but loop ended)
    final_disc, _, _ = _get_max_ap_discrepancy(signs)
    if final_disc < best_disc:
        best_disc = final_disc
        best_signs = signs.copy()

    return best_signs.tolist()

# EVOLVE-BLOCK-END
