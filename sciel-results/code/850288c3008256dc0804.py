import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1,n]."""
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
    """Build list of all arithmetic progressions with length >= 2."""
    aps = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                ap_lengths.append(len(sub_indices))
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    return aps, ap_membership, ap_lengths

def _compute_ap_extrema(aps, signs, ap_lengths):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        _update_single_ap(ap, signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
        
    return max_sums, min_sums

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums, length):
    """Update max/min subarray sum for a single AP."""
    if length < 50:
        max_end = 0
        max_sum = 0
        min_end = 0
        min_sum = 0
        
        for idx in ap:
            v = signs[idx]
            max_end = max(0, max_end + v)
            if max_end > max_sum:
                max_sum = max_end
            
            min_end = max(0, min_end - v)
            if min_end > min_sum:
                min_sum = min_end
                
        max_sums[ap_idx] = max_sum
        min_sums[ap_idx] = -min_sum
    else:
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        
        max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _find_worst_ap(max_sums, min_sums):
    """Find the AP with maximum absolute discrepancy."""
    if len(max_sums) == 0:
        return -1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    if max_pos >= min_neg:
        return int(np.argmax(max_sums))
    else:
        return int(np.argmin(min_sums))

def _find_best_flip_in_ap(ap, signs, ap_membership, aps, max_sums, min_sums, ap_lengths):
    """Find the best position to flip within a specific AP to reduce discrepancy."""
    n = len(signs)
    best_idx = -1
    best_disc = float('inf')
    
    # Save current state
    old_max = max_sums.copy()
    old_min = min_sums.copy()
    
    for pos in ap:
        old_val = signs[pos]
        signs[pos] = -old_val
        
        for ap_idx in ap_membership[pos]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_idx = pos
        
        # Restore
        signs[pos] = old_val
        for ap_idx in ap_membership[pos]:
            max_sums[ap_idx] = old_max[ap_idx]
            min_sums[ap_idx] = old_min[ap_idx]
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
    
    return best_idx, best_disc

def _greedy_construct(n, aps, ap_membership, ap_lengths, seed=0):
    """Greedy construction: pick each sign to minimize current max discrepancy."""
    rng = random.Random(seed)
    signs = [0] * n
    max_sums = np.zeros(len(aps), dtype=np.int32)
    min_sums = np.zeros(len(aps), dtype=np.int32)
    
    # Initialize all to 0, then greedily assign
    # Start with random initialization
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 256
    
    aps, ap_membership, ap_lengths = _build_ap_structure(n)
    
    candidates = []
    
    # Thue-Morse sequence
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)
    
    # Multiple random seeds
    for seed in range(30):
        rng = random.Random(seed)
        candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Greedy-like: start with all +1, then flip to reduce
    greedy1 = [1] * n
    candidates.append(greedy1)
    
    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)
    
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline - 15:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    # Simulated annealing with multiple restarts
    rng = random.Random(42)
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        if restart > 0:
            # Restart from a perturbed version of best
            signs = best_signs[:]
            # Perturb 10% of positions
            for _ in range(n // 10):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
            max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
            current_disc = _get_max_disc(max_sums, min_sums)
        else:
            # Continue from current best
            signs = best_signs[:]
            max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
            current_disc = _get_max_disc(max_sums, min_sums)
        
        temp = 3.0
        cool_rate = 0.995
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            # Targeted approach: find worst AP and try flipping positions within it
            worst_ap_idx = _find_worst_ap(max_sums, min_sums)
            if worst_ap_idx >= 0:
                worst_ap = aps[worst_ap_idx]
                # Try flipping each position in the worst AP
                best_flip_idx, best_flip_disc = _find_best_flip_in_ap(
                    worst_ap, signs, ap_membership, aps, max_sums, min_sums, ap_lengths
                )
                
                if best_flip_idx >= 0 and best_flip_disc < current_disc:
                    # Apply the best flip
                    old_val = signs[best_flip_idx]
                    signs[best_flip_idx] = -old_val
                    for ap_idx in ap_membership[best_flip_idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                    current_disc = best_flip_disc
                    no_improve_count = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    # No improvement from targeted approach, try random flip
                    idx = rng.randint(0, n - 1)
                    old_val = signs[idx]
                    signs[idx] = -old_val
                    for ap_idx in ap_membership[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                    new_disc = _get_max_disc(max_sums, min_sums)
                    
                    if new_disc < current_disc:
                        current_disc = new_disc
                        no_improve_count = 0
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        delta = new_disc - current_disc
                        if delta > 0 and temp > 0.05:
                            accept_prob = math.exp(-delta / temp)
                            if rng.random() < accept_prob:
                                current_disc = new_disc
                            else:
                                signs[idx] = old_val
                                for ap_idx in ap_membership[idx]:
                                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                        else:
                            signs[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
            else:
                no_improve_count += 1
            
            temp *= cool_rate
        
        # Final polish: strict hill climbing
        if time.time() < deadline - 2:
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                improved = False
                for idx in range(n):
                    old_val = signs[idx]
                    signs[idx] = -old_val
                    
                    for ap_idx in ap_membership[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                    
                    new_disc = _get_max_disc(max_sums, min_sums)
                    
                    if new_disc < current_disc:
                        current_disc = new_disc
                        improved = True
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, ap_lengths[ap_idx])
                
                if not improved:
                    break
    
    # Final verification
    final_disc = _max_ap_discrepancy(best_signs)
    if final_disc < best_disc:
        best_disc = final_disc
    
    return best_signs
