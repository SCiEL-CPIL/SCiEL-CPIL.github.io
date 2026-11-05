import time
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    mid = n // 2  # 23
    
    # Split into two halves
    A_dens = denominators[:mid]  # 23 items
    B_dens = denominators[mid:]  # 24 items
    
    def gen_subset_sums(dens):
        """Generate all subset sums as numpy array of floats"""
        m = len(dens)
        # Use iterative doubling
        sums = np.array([0.0])
        for d in dens:
            w = 1.0 / d
            sums = np.concatenate([sums, sums + w])
        return sums
    
    # Generate subset sums for A and B
    sums_A = gen_subset_sums(A_dens)
    sums_B = gen_subset_sums(B_dens)
    
    # Sort A sums
    sums_A_sorted = np.sort(sums_A)
    
    # For each B sum, find the largest A sum such that A_sum + B_sum < 1
    # i.e., A_sum < 1 - B_sum
    # Use searchsorted to find insertion point, then take element before it
    
    # Vectorized: for each b in sums_B, find idx = searchsorted(sums_A_sorted, 1.0 - b, side='right') - 1
    # But we need to handle the case where idx < 0 (no valid A sum)
    
    targets = 1.0 - sums_B
    # Find the largest A sum < target for each target
    # searchsorted with side='right' gives the index where target would be inserted to maintain order
    # so the element at idx-1 is the largest element < target
    idxs = np.searchsorted(sums_A_sorted, targets, side='right') - 1
    
    # Filter out invalid indices
    valid = idxs >= 0
    if not np.any(valid):
        # Fallback: just pick the largest single item
        best_idx = np.argmax(sums_A_sorted)
        a_sum = sums_A_sorted[best_idx]
        b_sum = 0.0
    else:
        valid_idxs = idxs[valid]
        valid_B = sums_B[valid]
        candidate_sums = sums_A_sorted[valid_idxs] + valid_B
        
        # Find the best candidate
        best_local = np.argmax(candidate_sums)
        a_sum = sums_A_sorted[valid_idxs[best_local]]
        b_sum = valid_B[best_local]
    
    total_sum_float = a_sum + b_sum
    
    # Now we need to recover which subsets these correspond to
    # This is tricky with the numpy approach. Let's use a different method.
    
    # Alternative: track masks. But with 8.4M and 16.7M entries, storing masks is expensive.
    
    # Let me use a different approach: iterate over B sums and for each, binary search A.
    # But do it more efficiently by only checking promising candidates.
    
    # Actually, let me just do the full search but track the best few candidates,
    # then verify with exact fractions.
    
    # Better: Let's generate subset sums WITH masks for the smaller half.
    # For A (23 items), 8.4M entries with 23-bit masks - feasible.
    # For B (24 items), 16.7M entries - let's just iterate.
    
    # Generate A sums with masks
    def gen_subset_sums_with_masks(dens):
        m = len(dens)
        sums = [0.0]
        masks = [0]
        for i, d in enumerate(dens):
            w = 1.0 / d
            bit = 1 << i
            new_sums = [s + w for s in sums]
            new_masks = [m | bit for m in masks]
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks
    
    # This might be slow for 8.4M entries in Python lists. Let me try a numpy-based mask approach.
    
    # Actually, let's just do the meet-in-the-middle with careful timing.
    # Generate A sums as numpy array (no masks needed if we can recover them differently)
    
    # Alternative approach: Since we just need to return indices, and the search space is large,
    # let's use a greedy + hill climbing approach with exact fractions.
    
    # Start with greedy: sort by 1/d descending, add if sum < 1
    items = [(1.0/d, i) for i, d in enumerate(denominators)]
    items.sort(reverse=True)
    
    selected = []
    total = Fraction(0)
    for frac_val, idx in items:
        if total + Fraction(1, denominators[idx]) < 1:
            selected.append(idx)
            total += Fraction(1, denominators[idx])
    
    best_deficit = 1 - total
    best_selected = list(selected)
    
    # Local search: try to improve by swapping or adding/removing
    # Time budget for local search
    search_deadline = start + 80
    
    # Try single swaps: remove one selected, add one unselected
    selected_set = set(selected)
    unselected = [i for i in range(n) if i not in selected_set]
    
    improved = True
    while improved and time.time() < search_deadline:
        improved = False
        # Try removing one and adding one
        for rem_idx in selected:
            rem_frac = Fraction(1, denominators[rem_idx])
            new_total = total - rem_frac
            # Try adding each unselected
            for add_idx in unselected:
                add_frac = Fraction(1, denominators[add_idx])
                if new_total + add_frac < 1:
                    new_deficit = 1 - (new_total + add_frac)
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        total = new_total + add_frac
                        selected.remove(rem_idx)
                        selected.append(add_idx)
                        selected_set.discard(rem_idx)
                        selected_set.add(add_idx)
                        unselected.remove(add_idx)
                        unselected.append(rem_idx)
                        improved = True
                        break
                if time.time() > search_deadline:
                    break
            if time.time() > search_deadline:
                break
        if time.time() > search_deadline:
            break
    
    # Also try adding items that fit
    if time.time() < search_deadline:
        for add_idx in unselected:
            add_frac = Fraction(1, denominators[add_idx])
            if total + add_frac < 1:
                new_deficit = 1 - (total + add_frac)
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    total += add_frac
                    selected.append(add_idx)
                    selected_set.add(add_idx)
                    unselected.remove(add_idx)
    
    # Verify
    final_total = sum(Fraction(1, denominators[i]) for i in selected)
    if final_total < 1:
        return selected
    else:
        # Fallback
        return [0]
