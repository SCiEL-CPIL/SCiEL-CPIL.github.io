import time
import math
import random
import numpy as np

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
            sub = signs_arr[r::d]
            cum = np.concatenate(([0], np.cumsum(sub)))
            min_prefix = np.minimum.accumulate(cum)
            max_prefix = np.maximum.accumulate(cum)
            max_sub = np.max(cum[1:] - min_prefix[:-1])
            min_sub = np.min(cum[1:] - max_prefix[:-1])
            current_max = max(abs(int(max_sub)), abs(int(min_sub)))
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
        cum = np.concatenate(([0], np.cumsum(vals)))
        min_prefix = np.minimum.accumulate(cum)
        max_prefix = np.maximum.accumulate(cum)
        max_sums[ap_idx] = np.max(cum[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum[1:] - max_prefix[:-1])
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
    cum = np.concatenate(([0], np.cumsum(vals)))
    min_prefix = np.minimum.accumulate(cum)
    max_prefix = np.maximum.accumulate(cum)
    max_sums[ap_idx] = np.max(cum[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum[1:] - max_prefix[:-1])

def _find_worst_ap(max_sums, min_sums):
    """Find the AP with the largest absolute discrepancy."""
    if len(max_sums) == 0:
        return -1
    max_pos_idx = int(np.argmax(max_sums))
    min_neg_idx = int(np.argmin(min_sums))
    if max_sums[max_pos_idx] >= -min_sums[min_neg_idx]:
        return max_pos_idx
    else:
        return min_neg_idx

def _targeted_descent(aps, ap_membership, signs, max_sums, min_sums, current_disc, rng, deadline):
    """Try to reduce discrepancy by targeting the worst AP."""
    if time.time() > deadline:
        return current_disc, False
    
    worst_ap_idx = _find_worst_ap(max_sums, min_sums)
    if worst_ap_idx < 0:
        return current_disc, False
    
    worst_ap = aps[worst_ap_idx]
    improved = False
    
    # Try flipping each element in the worst AP
    for pos in worst_ap:
        if time.time() > deadline:
            break
        
        old_val = signs[pos]
        signs[pos] = -old_val
        
        # Update all APs that include this position
        for ap_idx in ap_membership[pos]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        if new_disc < current_disc:
            current_disc = new_disc
            improved = True
        else:
            # Revert
            signs[pos] = old_val
            for ap_idx in ap_membership[pos]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
    
    return current_disc, improved

def _greedy_init(n):
    """Greedy initialization: build signs one by one, choosing the sign that minimizes current max AP discrepancy."""
    signs = [0] * n
    
    # Precompute which APs each position belongs to (for incremental tracking)
    # We'll track partial sums for APs that have been started
    
    # For greedy, we'll just use a simple heuristic: Thue-Morse as base, then refine
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    return signs

def _multi_start_sa(aps, ap_membership, n, deadline, seed_offset=0):
    """Run simulated annealing from a random start."""
    rng = random.Random(42 + seed_offset)
    
    # Random initialization
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    if best_disc == 1:
        return best_signs, best_disc
    
    temp = 3.0
    cool_rate = 0.9995
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 1:
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        new_val = -old_val
        
        signs[idx] = new_val
        
        for ap_idx in ap_membership[idx]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        delta = new_disc - current_disc
        if delta < 0 or (temp > 0.1 and rng.random() < math.exp(-delta / temp)):
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
            break
        
        # Targeted descent when stuck
        if no_improve > max_no_improve:
            current_disc, improved = _targeted_descent(
                aps, ap_membership, signs, max_sums, min_sums, current_disc, rng, deadline
            )
            if improved:
                no_improve = 0
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                # Random perturbation
                for _ in range(3):
                    pidx = rng.randint(0, n - 1)
                    signs[pidx] = -signs[pidx]
                max_sums, min_sums = _compute_ap_extrema(aps, signs)
                current_disc = _get_max_disc(max_sums, min_sums)
                no_improve = 0
                temp = max(temp, 1.5)
    
    return best_signs, best_disc

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin for 90s limit
    
    n = 256
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)
    
    # Walsh-Hadamard
    for k in [1, 2, 3, 4]:
        wh = [1 if (i >> k) & 1 == 0 else -1 for i in range(n)]
        candidates.append(wh)
    
    # Random seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
    
    # Sine waves
    for freq in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)
    
    # Evaluate candidates and pick best few
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 30:
            break
        disc = _max_ap_discrepancy(c)
        scored_candidates.append((disc, c))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_signs = scored_candidates[0][1][:]
    best_disc = scored_candidates[0][0]
    
    if best_disc == 1:
        return best_signs
    
    # Run multi-start SA from top 3 candidates and a couple random starts
    top_starts = [c for _, c in scored_candidates[:3]]
    rng_init = random.Random(123)
    for _ in range(2):
        top_starts.append([rng_init.choice((-1, 1)) for _ in range(n)])
    
    for start_idx, start_signs in enumerate(top_starts):
        if time.time() > deadline - 5:
            break
        
        # Run SA from this start
        sa_signs, sa_disc = _multi_start_sa(aps, ap_membership, n, deadline, seed_offset=start_idx)
        
        if sa_disc < best_disc:
            best_disc = sa_disc
            best_signs = sa_signs[:]
        
        if best_disc == 1:
            break
    
    # Final polish: run targeted descent on the best result
    if time.time() < deadline - 2 and best_disc > 1:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_extrema(aps, signs)
        current_disc = _get_max_disc(max_sums, min_sums)
        
        rng = random.Random(999)
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            current_disc, improved = _targeted_descent(
                aps, ap_membership, signs, max_sums, min_sums, current_disc, rng, deadline
            )
            if not improved:
                break
        
        final_disc = _get_max_disc(max_sums, min_sums)
        if final_disc < best_disc:
            best_disc = final_disc
            best_signs = signs[:]
    
    return best_signs
