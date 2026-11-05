import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1..n]."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    max_abs = 1  # singletons
    
    # We iterate over step d
    for d in range(1, n):
        # For each residue class mod d
        for r in range(d):
            if r >= n:
                break
            # Get indices
            indices = np.arange(r, n, d)
            if len(indices) < 2:
                continue
            sub = signs_arr[indices]
            
            # Compute max subarray sum
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

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """Update max/min subarray sum for a single AP."""
    # Use numpy for speed
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _compute_ap_extrema(aps, signs):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        _update_single_ap(ap, signs, ap_idx, max_sums, min_sums)
        
    return max_sums, min_sums

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    # max_sums are positive max sums
    # min_sums are negative min sums (so -min_sums are positive abs min sums)
    # We want max( max(max_sums), max(-min_sums) )
    max_pos = int(np.max(max_sums))
    max_neg = int(-np.min(min_sums))
    return max(max_pos, max_neg, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership, ap_lengths = _build_ap_structure(n)
    n_aps = len(aps)
    
    # Precompute sorted membership lists for faster iteration
    for i in range(n):
        ap_membership[i].sort()
        
    # Candidates
    candidates = []
    # Thue-Morse
    signs_tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_alt)
    
    # Random
    for seed in range(10):
        rng_c = random.Random(seed)
        candidates.append([rng_c.choice((-1, 1)) for _ in range(n)])
        
    # All +1
    candidates.append([1] * n)
    
    # Evaluate candidates to find best start
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 20:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
            
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
        
    # Phase 1: Targeted Descent (Single Flips)
    # We try to flip a sign in the worst AP to reduce its discrepancy.
    phase1_deadline = deadline - 20
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < phase1_deadline and no_improve_count < max_no_improve:
        # Find worst AP
        worst_ap_idx = None
        worst_val = 0
        
        max_pos_idx = int(np.argmax(max_sums))
        max_pos_val = int(max_sums[max_pos_idx])
        
        min_neg_idx = int(np.argmin(min_sums))
        min_neg_val = int(-min_sums[min_neg_idx])
        
        if max_pos_val >= min_neg_val:
            worst_ap_idx = max_pos_idx
            worst_val = max_pos_val
        else:
            worst_ap_idx = min_neg_idx
            worst_val = min_neg_val
            
        if worst_val <= 1:
            break
            
        ap = aps[worst_ap_idx]
        
        # Try flipping each element in this AP
        # If AP is long, sample; if short, try all
        if len(ap) <= 20:
            positions_to_try = list(range(len(ap)))
        else:
            positions_to_try = set()
            positions_to_try.add(0)
            positions_to_try.add(len(ap) - 1)
            positions_to_try.add(len(ap) // 2)
            if len(ap) > 4:
                positions_to_try.add(len(ap) // 4)
                positions_to_try.add(3 * len(ap) // 4)
            # Add some random
            for _ in range(10):
                positions_to_try.add(rng.randint(0, len(ap) - 1))
            positions_to_try = list(positions_to_try)
            
        best_elem = None
        best_improvement = 0
        best_new_disc = current_disc
        
        for pos in positions_to_try:
            if time.time() > phase1_deadline:
                break
            elem_idx = ap[pos]
            old_val = signs[elem_idx]
            signs[elem_idx] = -old_val
            
            # Update affected APs
            for ap_idx in ap_membership[elem_idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            improvement = current_disc - new_disc
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_elem = elem_idx
                best_new_disc = new_disc
                # Revert to try next
                signs[elem_idx] = old_val
                for ap_idx in ap_membership[elem_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            else:
                signs[elem_idx] = old_val
                for ap_idx in ap_membership[elem_idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                    
        if best_elem is not None:
            # Apply the best flip
            signs[best_elem] = -signs[best_elem]
            for ap_idx in ap_membership[best_elem]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            current_disc = best_new_disc
            no_improve_count = 0
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            no_improve_count += 1
            # Random kick
            idx = rng.randint(0, n - 1)
            signs[idx] = -signs[idx]
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            current_disc = _get_max_disc(max_sums, min_sums)
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
                
    # Phase 2: Simulated Annealing with Pair Flips
    rng_sa = random.Random(123)
    temp = 2.0
    cool_rate = 0.99
    no_improve_sa = 0
    max_no_improve_sa = 100
    
    while time.time() < deadline - 5 and no_improve_sa < max_no_improve_sa:
        # Decide: single flip or pair flip?
        if rng_sa.random() < 0.3:
            # Pair flip
            idx1 = rng_sa.randint(0, n - 1)
            idx2 = rng_sa.randint(0, n - 1)
            while idx2 == idx1:
                idx2 = rng_sa.randint(0, n - 1)
            
            old_val1 = signs[idx1]
            old_val2 = signs[idx2]
            
            signs[idx1] = -old_val1
            signs[idx2] = -old_val2
            
            # Update union of affected APs
            affected_aps = set(ap_membership[idx1])
            affected_aps.update(ap_membership[idx2])
            
            for ap_idx in affected_aps:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            delta = new_disc - current_disc
            
            if delta < 0 or rng_sa.random() < math.exp(-delta / max(temp, 0.01)):
                current_disc = new_disc
                no_improve_sa = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                signs[idx1] = old_val1
                signs[idx2] = old_val2
                for ap_idx in affected_aps:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                no_improve_sa += 1
        else:
            # Single flip
            idx = rng_sa.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            affected_aps = ap_membership[idx]
            for ap_idx in affected_aps:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                
            new_disc = _get_max_disc(max_sums, min_sums)
            delta = new_disc - current_disc
            
            if delta < 0 or rng_sa.random() < math.exp(-delta / max(temp, 0.01)):
                current_disc = new_disc
                no_improve_sa = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                signs[idx] = old_val
                for ap_idx in affected_aps:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                no_improve_sa += 1
                
        temp *= cool_rate
        
    # Phase 3: Final Local Search (Single Flips)
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old_val = signs[i]
                signs[i] = -old_val
                for ap_idx in ap_membership[i]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                new_disc = _get_max_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs[:]
                else:
                    signs[i] = old_val
                    for ap_idx in ap_membership[i]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                        
    return best_signs
