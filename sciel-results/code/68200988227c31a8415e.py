import time
from fractions import Fraction
import bisect
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Split into two halves for meet-in-the-middle
    n1 = n // 2  # 23
    n2 = n - n1  # 24

    def gen_sums_with_masks(dens):
        """Generate all subset sums with corresponding masks using iterative doubling."""
        m = len(dens)
        sums = np.array([0.0])
        masks = np.array([0], dtype=np.int64)
        for i in range(m):
            val = 1.0 / dens[i]
            bit = 1 << i
            sums = np.concatenate([sums, sums + val])
            masks = np.concatenate([masks, masks | bit])
        return sums, masks

    # Generate subset sums for both halves
    sums1, masks1 = gen_sums_with_masks(denominators[:n1])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    sums2, masks2 = gen_sums_with_masks(denominators[n1:])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Sort sums2 with corresponding masks
    order = np.argsort(sums2)
    sums2_sorted = sums2[order]
    masks2_sorted = masks2[order]
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # For each s1 in sums1, find the best s2 such that s1 + s2 < 1
    # We want to maximize s1 + s2 subject to s1 + s2 < 1
    # Equivalently, for each s1, find the largest s2 < 1 - s1
    
    valid1 = sums1 < 1.0
    s1_valid = sums1[valid1]
    masks1_valid = masks1[valid1]
    targets = 1.0 - s1_valid
    
    # For each target, find the rightmost s2 < target
    # np.searchsorted with side='left' gives insertion point
    # We want the index before that (largest value < target)
    pos = np.searchsorted(sums2_sorted, targets, side='left')
    
    # Handle pos=0 case: no valid s2 (shouldn't happen since 0 is in sums2)
    valid_pos = pos > 0
    if not np.any(valid_pos):
        return greedy_fallback(denominators)
    
    pos_valid = pos[valid_pos]
    s2_candidates = sums2_sorted[pos_valid - 1]
    m2_candidates = masks2_sorted[pos_valid - 1]
    s1_candidates = s1_valid[valid_pos]
    m1_candidates = masks1_valid[valid_pos]
    
    totals = s1_candidates + s2_candidates
    valid_total = totals < 1.0
    if not np.any(valid_total):
        return greedy_fallback(denominators)
    
    totals_valid = totals[valid_total]
    s1_final = s1_candidates[valid_total]
    s2_final = s2_candidates[valid_total]
    m1_final = m1_candidates[valid_total]
    m2_final = m2_candidates[valid_total]
    
    deficits = 1.0 - totals_valid
    best_idx = np.argmin(deficits)
    best_mask1 = int(m1_final[best_idx])
    best_mask2 = int(m2_final[best_idx])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Convert masks to indices
    selected = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected.append(n1 + i)

    # Verify with Fraction to ensure strict < 1 and get exact deficit
    if not selected:
        return greedy_fallback(denominators)

    total_frac = Fraction()
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
    
    if total_frac >= 1:
        # Float precision issue - try nearby candidates
        # Check a few more candidates from the top deficits
        sorted_deficits = np.argsort(deficits)
        for check_idx in sorted_deficits[:10]:
            m1 = int(m1_final[check_idx])
            m2 = int(m2_final[check_idx])
            sel = []
            for i in range(n1):
                if m1 & (1 << i):
                    sel.append(i)
            for i in range(n2):
                if m2 & (1 << i):
                    sel.append(n1 + i)
            if not sel:
                continue
            tf = Fraction()
            for idx in sel:
                tf += Fraction(1, denominators[idx])
            if tf < 1:
                return sel
        
        return greedy_fallback(denominators)

    return selected

def greedy_fallback(denominators):
    """Simple greedy: add largest available term if sum < 1."""
    selected = []
    total = Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
