import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256

    # Build all APs with length >= 2, store as numpy arrays for fast access
    aps = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = np.arange(r, n, d, dtype=np.int32)
            if len(sub_indices) >= 2:
                aps.append(sub_indices)
    
    num_aps = len(aps)
    ap_lengths = np.array([len(ap) for ap in aps], dtype=np.int32)

    def _compute_all_ap_sums(signs):
        """Compute max subarray sum and min subarray sum for each AP."""
        max_sums = np.zeros(num_aps, dtype=np.int32)
        min_sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(num_aps):
            ap = aps[i]
            vals = signs[ap]
            cum = np.cumsum(vals)
            cum_ext = np.concatenate(([0], cum))
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            max_sums[i] = np.max(cum_ext - min_prefix)
            min_sums[i] = np.min(cum_ext - max_prefix)
        return max_sums, min_sums

    def _get_max_ap_discrepancy(max_sums, min_sums):
        if num_aps == 0:
            return 1
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _find_worst_ap_indices(max_sums, min_sums, k=5):
        """Find indices of the k worst APs."""
        abs_disc = np.maximum(max_sums, -min_sums)
        if num_aps <= k:
            return np.arange(num_aps)
        return np.argsort(-abs_disc)[:k]

    # --- Phase 1: Generate diverse initial candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # Block patterns with various block sizes
    for block_size in [2, 3, 4, 6, 8]:
        direct = [0] * n
        for i in range(n):
            block = (i // block_size) % 2
            pos = i % block_size
            half = block_size // 2
            if block == 0:
                direct[i] = 1 if pos < half else -1
            else:
                direct[i] = -1 if pos < half else 1
        candidates.append(direct)

    # Random candidates
    for s in range(30):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = np.array(candidates[0], dtype=np.int32)
    best_disc = float('inf')

    for cand in candidates:
        if time.time() > deadline - 40:
            break
        signs_arr = np.array(cand, dtype=np.int32)
        max_sums, min_sums = _compute_all_ap_sums(signs_arr)
        d = _get_max_ap_discrepancy(max_sums, min_sums)
        if d < best_disc:
            best_disc = d
            best_signs = signs_arr.copy()

    # --- Phase 3: Greedy Worst-AP Repair ---
    signs = best_signs.copy()
    max_sums, min_sums = _compute_all_ap_sums(signs)
    current_disc = _get_max_ap_discrepancy(max_sums, min_sums)

    # Precompute for each index which APs it belongs to
    # For efficiency, we'll maintain a mapping
    ap_membership = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            ap_membership[idx].append(ap_idx)

    def _update_single_ap(ap_idx):
        """Recompute sums for a single AP."""
        ap = aps[ap_idx]
        vals = signs[ap]
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sub = int(np.max(cum_ext - min_prefix))
        min_sub = int(np.min(cum_ext - max_prefix))
        return max_sub, min_sub

    rng = random.Random(12345)
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Find worst APs
        worst_indices = _find_worst_ap_indices(max_sums, min_sums, k=3)
        
        # For each worst AP, try to fix it by flipping elements
        found_improvement = False
        
        for ap_idx in worst_indices:
            if time.time() > deadline - 10:
                break
            if found_improvement:
                break
                
            ap = aps[ap_idx]
            # Try flipping each element in the worst AP
            for pos in range(len(ap)):
                if time.time() > deadline - 10:
                    break
                if found_improvement:
                    break
                    
                idx = ap[pos]
                # Flip this element
                signs[idx] = -signs[idx]
                
                # Update all APs that contain this index
                affected_aps = ap_membership[idx]
                for a_idx in affected_aps:
                    max_sums[a_idx], min_sums[a_idx] = _update_single_ap(a_idx)
                
                new_disc = _get_max_ap_discrepancy(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    found_improvement = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[idx] = -signs[idx]
                    for a_idx in affected_aps:
                        max_sums[a_idx], min_sums[a_idx] = _update_single_ap(a_idx)
        
        if not found_improvement:
            no_improve += 1
            # If stuck, try a random perturbation
            if no_improve % 50 == 0:
                # Flip 3 random elements
                for _ in range(3):
                    idx = rng.randint(0, n - 1)
                    signs[idx] = -signs[idx]
                    for a_idx in ap_membership[idx]:
                        max_sums[a_idx], min_sums[a_idx] = _update_single_ap(a_idx)
                current_disc = _get_max_ap_discrepancy(max_sums, min_sums)

    # --- Phase 4: Final Polish with Single Flips ---
    if time.time() < deadline - 3:
        signs = best_signs.copy()
        max_sums, min_sums = _compute_all_ap_sums(signs)
        current_disc = _get_max_ap_discrepancy(max_sums, min_sums)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 3:
                    break
                signs[idx] = -signs[idx]
                for a_idx in ap_membership[idx]:
                    max_sums[a_idx], min_sums[a_idx] = _update_single_ap(a_idx)
                new_disc = _get_max_ap_discrepancy(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs.copy()
                else:
                    signs[idx] = -signs[idx]
                    for a_idx in ap_membership[idx]:
                        max_sums[a_idx], min_sums[a_idx] = _update_single_ap(a_idx)

    return list(best_signs)

