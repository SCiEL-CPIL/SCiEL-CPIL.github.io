import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or numpy array of +1/-1.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            cum = np.cumsum(sub)
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def _build_ap_structure(n):
    """
    Precompute all arithmetic progressions (start, step, length>=2) 
    and index their elements for fast incremental discrepancy tracking.
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    return aps, ap_membership

def _compute_ap_extrema(aps, signs):
    """
    For each AP, compute the maximum subarray sum and minimum subarray sum.
    """
    max_sums = []
    min_sums = []
    for ap in aps:
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        
        max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
        
        max_sums.append(max_sub)
        min_sums.append(min_sub)
    
    return max_sums, min_sums

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """
    Recompute max/min subarray sums for a single AP.
    """
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _get_max_disc(max_sums, min_sums):
    """Get the maximum absolute discrepancy from precomputed sums."""
    if not max_sums:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _find_worst_aps(max_sums, min_sums, k=5):
    """
    Find indices of APs with the highest absolute discrepancy.
    """
    abs_discs = np.maximum(np.abs(max_sums), np.abs(min_sums))
    # Get top k indices
    if len(abs_discs) == 0:
        return []
    top_indices = np.argsort(abs_discs)[::-1]
    return top_indices[:k].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    aps, ap_membership = _build_ap_structure(n)
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    tm = []
    for i in range(n):
        cnt = bin(i).count('1')
        tm.append(1 if cnt % 2 == 0 else -1)
    candidates.append(tm)
    
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
        
    # Evaluate and pick best
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 15:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]

    # 2. SAT-style Local Search: Greedy Worst-AP Repair
    signs = best_signs[:]
    
    # Compute initial AP extrema
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    rng = random.Random(42)
    
    # WalkSAT parameters
    # We prioritize fixing the "worst" APs.
    # Strategy:
    # 1. Identify top K worst APs.
    # 2. For each, try to flip a variable within it to reduce its discrepancy.
    # 3. If no single flip improves the global max, or if we get stuck, 
    #    perform a random perturbation on the variables involved in the worst APs.
    
    max_iters = 20000
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 2 and iter_count < max_iters:
        iter_count += 1
        
        # Identify worst APs
        worst_aps = _find_worst_aps(max_sums, min_sums, k=3)
        
        if not worst_aps:
            break
            
        improved = False
        
        # Try to fix one of the worst APs
        for ap_idx in worst_aps:
            if time.time() > deadline - 2:
                break
                
            ap = aps[ap_idx]
            # Try flipping each element in this AP
            # Heuristic: flip the element that contributes most to the current discrepancy
            # For simplicity and robustness, we try random elements in the AP first,
            # then if that fails, we try all elements in the AP (if small) or a random subset.
            
            # To be fast, we only try a subset of flips per AP per iteration
            # Let's try up to 5 random flips within this specific AP
            trial_flips = min(5, len(ap))
            indices_to_try = rng.sample(ap, trial_flips) if len(ap) > trial_flips else ap[:]
            
            for idx in indices_to_try:
                old_val = signs[idx]
                new_val = -old_val
                
                # Flip
                signs[idx] = new_val
                
                # Update affected APs
                affected_aps = ap_membership[idx]
                for a_idx in affected_aps:
                    _update_single_ap(aps[a_idx], signs, a_idx, max_sums, min_sums)
                
                new_disc = _get_max_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    # Accept improvement
                    current_disc = new_disc
                    improved = True
                    no_improve_count = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                    # Break out of inner loops to re-evaluate worst APs
                    break
                else:
                    # Revert
                    signs[idx] = old_val
                    for a_idx in affected_aps:
                        _update_single_ap(aps[a_idx], signs, a_idx, max_sums, min_sums)
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            
            # If stuck, do a "perturbation" phase:
            # Randomly flip some variables that are part of the current worst APs
            # This helps escape local minima where single flips don't help.
            if no_improve_count > 20:
                # Pick a random worst AP and flip 2-3 random elements in it
                target_ap_idx = rng.choice(worst_aps)
                target_ap = aps[target_ap_idx]
                
                num_flips = rng.randint(2, 4)
                if len(target_ap) > num_flips:
                    flip_indices = rng.sample(target_ap, num_flips)
                else:
                    flip_indices = target_ap[:]
                    
                for idx in flip_indices:
                    signs[idx] = -signs[idx]
                    for a_idx in ap_membership[idx]:
                        _update_single_ap(aps[a_idx], signs, a_idx, max_sums, min_sums)
                        
                current_disc = _get_max_disc(max_sums, min_sums)
                
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
                    
                no_improve_count = 0
                # Reheat? No, just reset counter.
                
    # Final verification
    final_disc = _max_ap_discrepancy(signs)
    if final_disc < best_disc:
        best_disc = final_disc
        best_signs = signs[:]
    
    return best_signs

if __name__ == "__main__":
    # Quick test
    signs = construct()
    print("Final discrepancy:", _max_ap_discrepancy(signs))
