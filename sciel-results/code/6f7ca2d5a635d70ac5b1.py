def construct():
    import time
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    from fractions import Fraction
    import numpy as np
    
    denominators = list(range(2, 49))  # 47 items
    n = len(denominators)
    
    # Split into two halves
    nA = n // 2  # 23
    nB = n - nA  # 24
    
    A_denoms = denominators[:nA]
    B_denoms = denominators[nA:]
    
    A_vals = [1.0 / d for d in A_denoms]
    B_vals = [1.0 / d for d in B_denoms]
    
    # Generate all subset sums for A
    A_sums = np.array([0.0])
    A_masks = np.array([0])
    for i, v in enumerate(A_vals):
        new_sums = A_sums + v
        new_masks = A_masks | (1 << i)
        A_sums = np.concatenate([A_sums, new_sums])
        A_masks = np.concatenate([A_masks, new_masks])
    
    # Generate all subset sums for B
    B_sums = np.array([0.0])
    B_masks = np.array([0])
    for i, v in enumerate(B_vals):
        new_sums = B_sums + v
        new_masks = B_masks | (1 << i)
        B_sums = np.concatenate([B_sums, new_sums])
        B_masks = np.concatenate([B_masks, new_masks])
    
    if time.time() > deadline:
        # Fallback to greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    # Sort B by sum
    sort_idx = np.argsort(B_sums)
    B_sums_sorted = B_sums[sort_idx]
    B_masks_sorted = B_masks[sort_idx]
    
    # For each a in A, find largest b < 1 - a
    targets = 1.0 - A_sums
    indices = np.searchsorted(B_sums_sorted, targets, side='left') - 1
    
    # Filter valid
    valid = indices >= 0
    if not np.any(valid):
        # Fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    valid_indices = np.where(valid)[0]
    best_a_idx = valid_indices
    best_b_idx = indices[valid_indices]
    
    candidate_sums = A_sums[best_a_idx] + B_sums_sorted[best_b_idx]
    
    # Find the maximum candidate sum < 1
    max_pos = np.argmax(candidate_sums)
    best_a_pos = best_a_idx[max_pos]
    best_b_pos = best_b_idx[max_pos]
    
    # Recover masks
    A_mask = A_masks[best_a_pos]
    B_mask = B_masks_sorted[best_b_pos]
    
    # Convert to indices
    selected = []
    for i in range(nA):
        if A_mask & (1 << i):
            selected.append(i)
    for i in range(nB):
        if B_mask & (1 << i):
            selected.append(nA + i)
    
    # Verify with exact fractions
    total = sum(Fraction(1, denominators[i]) for i in selected)
    if total < 1:
        return selected
    else:
        # Float error, try next best candidates
        # Sort candidates by sum descending, try top 100
        order = np.argsort(candidate_sums)[::-1]
        for pos in order[:100]:
            a_idx = best_a_idx[pos]
            b_idx = best_b_idx[pos]
            A_mask = A_masks[a_idx]
            B_mask = B_masks_sorted[b_idx]
            sel = []
            for i in range(nA):
                if A_mask & (1 << i):
                    sel.append(i)
            for i in range(nB):
                if B_mask & (1 << i):
                    sel.append(nA + i)
            total = sum(Fraction(1, denominators[i]) for i in sel)
            if total < 1:
                return sel
    
    # Fallback to greedy
    selected, total = [], Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
