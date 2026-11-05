import time
import random
import numpy as np
import math

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1..n]."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 1
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            m = (n - 1 - r) // d + 1
            if m <= 1:
                continue
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            cum = np.concatenate(([0], np.cumsum(sub)))
            # Max subarray sum
            running_min = 0
            running_max = 0
            max_sum = 0
            min_sum = 0
            for i in range(1, m + 1):
                val = cum[i]
                if val < running_min:
                    running_min = val
                if val > running_max:
                    running_max = val
                diff_max = val - running_min
                diff_min = val - running_max
                if diff_max > max_sum:
                    max_sum = diff_max
                if diff_min < min_sum:
                    min_sum = diff_min
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
    return max_abs

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2 and membership info."""
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
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])
    return max_sums, min_sums

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """Update max/min subarray sum for a single AP in-place."""
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _find_worst_ap_and_element(aps, signs, max_sums, min_sums):
    """Find the worst AP and the element within it that would most reduce discrepancy if flipped."""
    n_aps = len(aps)
    if n_aps == 0:
        return -1, -1
    
    # Find the AP with the largest absolute discrepancy
    worst_ap = -1
    worst_disc = 0
    for ap_idx in range(n_aps):
        disc = max(abs(int(max_sums[ap_idx])), abs(int(min_sums[ap_idx])))
        if disc > worst_disc:
            worst_disc = disc
            worst_ap = ap_idx
    
    if worst_ap == -1:
        return -1, -1
    
    ap = aps[worst_ap]
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.concatenate(([0], np.cumsum(vals)))
    
    # Find the position in this AP that, if flipped, would most reduce the max subarray sum
    # We try flipping each position and see which reduces the AP's discrepancy the most
    best_pos = -1
    best_improvement = 0
    
    for pos in range(len(ap)):
        # Simulate flipping position pos
        old_val = vals[pos]
        new_val = -old_val
        
        # Recompute cum with flip
        cum_new = cum.copy()
        # Update cum from pos onward
        diff = new_val - old_val
        cum_new[pos+1:] += diff
        
        # Compute max subarray sum
        min_prefix = np.minimum.accumulate(cum_new)
        max_prefix = np.maximum.accumulate(cum_new)
        max_sum_new = np.max(cum_new[1:] - min_prefix[:-1])
        min_sum_new = np.min(cum_new[1:] - max_prefix[:-1])
        
        disc_new = max(abs(int(max_sum_new)), abs(int(min_sum_new)))
        improvement = worst_disc - disc_new
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_pos = pos
    
    if best_pos == -1:
        return worst_ap, -1
    
    return worst_ap, ap[best_pos]

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # Thue-Morse initialization
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    # If we already have disc 1, we are done
    if best_disc == 1:
        return best_signs

    # Phase 1: Targeted descent - directly attack the worst AP
    targeted_iterations = 0
    max_targeted = 5000
    
    while time.time() < deadline and targeted_iterations < max_targeted:
        targeted_iterations += 1
        
        # Find the worst AP and the element to flip
        worst_ap_idx, flip_idx = _find_worst_ap_and_element(aps, signs, max_sums, min_sums)
        
        if flip_idx == -1:
            # No improvement possible by flipping within the worst AP
            break
        
        old_val = signs[flip_idx]
        new_val = -old_val
        
        signs[flip_idx] = new_val
        
        for ap_idx in ap_membership[flip_idx]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
        else:
            # Revert
            signs[flip_idx] = old_val
            for ap_idx in ap_membership[flip_idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            break  # No improvement from targeted descent, move to SA
        
        if best_disc == 1:
            return best_signs
    
    # Phase 2: Simulated Annealing with random flips
    temp = 1.0
    cool_rate = 0.9995
    no_improve = 0
    sa_iterations = 0
    max_sa = 500000
    
    while time.time() < deadline and sa_iterations < max_sa:
        sa_iterations += 1
        
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        new_val = -old_val
        
        signs[idx] = new_val
        
        for ap_idx in ap_membership[idx]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        delta = new_disc - current_disc
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            signs[idx] = old_val
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            no_improve += 1
            
        temp *= cool_rate
        
        if best_disc == 1:
            return best_signs
        
        # Periodic targeted descent during SA
        if sa_iterations % 1000 == 0 and time.time() < deadline - 5:
            for _ in range(100):
                if time.time() >= deadline - 5:
                    break
                worst_ap_idx, flip_idx = _find_worst_ap_and_element(aps, signs, max_sums, min_sums)
                if flip_idx == -1:
                    break
                old_val = signs[flip_idx]
                new_val = -old_val
                signs[flip_idx] = new_val
                for ap_idx in ap_membership[flip_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                new_disc = _get_max_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs[:]
                else:
                    signs[flip_idx] = old_val
                    for ap_idx in ap_membership[flip_idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                    break
    
    # Phase 3: Final polish - targeted descent with more iterations
    if time.time() < deadline - 2:
        polish_iterations = 0
        max_polish = 20000
        while time.time() < deadline - 2 and polish_iterations < max_polish:
            polish_iterations += 1
            worst_ap_idx, flip_idx = _find_worst_ap_and_element(aps, signs, max_sums, min_sums)
            if flip_idx == -1:
                break
            old_val = signs[flip_idx]
            new_val = -old_val
            signs[flip_idx] = new_val
            for ap_idx in ap_membership[flip_idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            new_disc = _get_max_disc(max_sums, min_sums)
            if new_disc < current_disc:
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                signs[flip_idx] = old_val
                for ap_idx in ap_membership[flip_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                break
    
    return best_signs
