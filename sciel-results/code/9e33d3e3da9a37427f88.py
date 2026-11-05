"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Fast Incremental Update."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_ap_structure(n):
    """
    Precompute the mapping from (d, i) to a flat index for APs,
    and the list of flat indices affected by flipping each position k.
    """
    # Total number of APs
    # For d=1, starts 0..0 (1 AP)
    # For d=2, starts 0..1 (2 APs)
    # ...
    # For d=n, starts 0..n-1 (n APs)
    total_aps = n * (n + 1) // 2
    
    # ap_ids[d_idx][i] = flat_index
    # d_idx = d - 1
    # i = start
    ap_ids = np.zeros((n, n), dtype=np.int32)
    
    flat_idx = 0
    for d in range(1, n + 1):
        for i in range(d):
            ap_ids[d - 1, i] = flat_idx
            flat_idx += 1
            
    # affected_ids[k] = array of flat indices of APs containing k
    # For a given k, the AP with step d containing k has start i = k % d.
    # So for each d in 1..n, the AP is (k % d, d).
    # We need to extract ap_ids[d-1, k % d] for all d.
    
    # We can precompute this for all k to avoid modulo ops in the loop?
    # Or just do it in the loop. n=256, so 256 modulo ops per step is negligible.
    
    return ap_ids, total_aps

def _init_ap_sums(signs, ap_ids, n):
    """
    Calculate initial sums for all APs.
    signs: array of +1/-1
    """
    s = np.asarray(signs, dtype=np.int16)
    total_aps = n * (n + 1) // 2
    ap_sums = np.zeros(total_aps, dtype=np.int16)
    
    for d in range(1, n + 1):
        d_idx = d - 1
        for i in range(d):
            flat_idx = ap_ids[d_idx, i]
            # Sum signs[i::d]
            # Using slice for speed
            ap_sums[flat_idx] = np.sum(s[i:n:d])
            
    return ap_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    rng = np.random.default_rng(42)
    
    ap_ids, total_aps = _precompute_ap_structure(n)
    
    best_signs = None
    best_disc = float('inf')
    
    # We will run multiple SA instances (restarting)
    # Each instance runs until time is up or T is too low.
    
    while time.time() < deadline:
        # Initialize random signs
        signs = rng.choice([-1, 1], size=n).astype(np.int16)
        ap_sums = _init_ap_sums(signs, ap_ids, n)
        current_max = int(np.max(np.abs(ap_sums)))
        
        if current_max < best_disc:
            best_disc = current_max
            best_signs = signs.copy()
        
        # SA Parameters
        # Delta in max_disc is integer.
        # Start T high enough to accept bad moves.
        T = 5.0 
        alpha = 0.9995 # Slow cooling to explore more
        min_T = 1e-3
        
        # Track stagnation
        last_improve_step = 0
        step = 0
        no_improve_limit = 5000 # If no improvement for 5k steps, restart this SA run
        
        # Precompute d_vals for fast modulo access
        d_vals = np.arange(1, n + 1)
        
        while time.time() < deadline:
            step += 1
            
            # Pick random index to flip
            k = rng.integers(0, n)
            
            # Calculate affected AP indices
            # For each d, start i = k % d
            starts = k % d_vals
            # Get flat indices
            # ap_ids is (n, n). d_vals is 1..n. So d_idx = d_vals - 1.
            affected_ids = ap_ids[d_vals - 1, starts]
            
            # Calculate delta
            # If we flip signs[k], the sum of each AP containing k changes by -2 * signs[k]
            delta = -2 * signs[k]
            
            # Apply change to ap_sums
            ap_sums[affected_ids] += delta
            
            # Calculate new max
            # np.max on 32k elements is fast
            new_max = int(np.max(np.abs(ap_sums)))
            
            # Metropolis Criterion
            if new_max < current_max:
                # Always accept improvement
                accept = True
            else:
                delta_disc = new_max - current_max
                if delta_disc == 0:
                    accept = True # Accept neutral moves to help exploration
                else:
                    # Accept with probability exp(-delta / T)
                    # Use log for stability if T is small? 
                    # delta is small integer, T is float.
                    if T > 1e-6:
                        prob = np.exp(-delta_disc / T)
                        accept = (rng.random() < prob)
                    else:
                        accept = False
            
            if accept:
                signs[k] = -signs[k]
                current_max = new_max
                
                if current_max < best_disc:
                    best_disc = current_max
                    best_signs = signs.copy()
                    last_improve_step = step
            else:
                # Revert change
                ap_sums[affected_ids] -= delta
            
            # Cool down
            T *= alpha
            
            # Check for stagnation or end of temperature
            if T < min_T or (step - last_improve_step > no_improve_limit):
                # Break out of this SA run to restart
                break
        
        # If time is up, the outer loop will also break
        if time.time() >= deadline:
            break

    if best_signs is None:
        # Fallback (should not happen)
        return [1] * n
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
