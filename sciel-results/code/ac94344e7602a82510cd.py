import time
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 83  # 7s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]
    
    # Split into two sets
    mid = n // 2
    A = reciprocals[:mid]
    B = reciprocals[mid:]
    
    # Generate all subset sums for A with masks
    sums_A = [(0.0, 0)]
    for i, w in enumerate(A):
        current_len = len(sums_A)
        for j in range(current_len):
            s, m = sums_A[j]
            sums_A.append((s + w, m | (1 << i)))
    
    # Generate all subset sums for B with masks
    sums_B = [(0.0, 0)]
    for i, w in enumerate(B):
        current_len = len(sums_B)
        for j in range(current_len):
            s, m = sums_B[j]
            sums_B.append((s + w, m | (1 << i)))
    
    # Sort sums_A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Two-pointer: find max sum_A + sum_B < 1.0
    best_sum = -1.0
    best_a_mask = 0
    best_b_mask = 0
    
    i = len(sums_A) - 1
    for j in range(len(sums_B)):
        b_sum, b_mask = sums_B[j]
        while i >= 0 and sums_A[i][0] + b_sum >= 1.0:
            i -= 1
        if i < 0:
            break
        a_sum, a_mask = sums_A[i]
        total = a_sum + b_sum
        if total > best_sum:
            best_sum = total
            best_a_mask = a_mask
            best_b_mask = b_mask
        if time.time() > deadline - 5:
            break
    
    # Reconstruct indices
    selected = []
    for i in range(mid):
        if best_a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if best_b_mask & (1 << i):
            selected.append(mid + i)
    
    if not selected:
        selected = [0]
    
    # Verify and polish with exact arithmetic
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    
    if total_frac >= 1:
        # Fix: remove elements until sum < 1, keeping the best deficit
        selected_set = set(selected)
        best_fixed = None
        best_deficit = None
        for k1 in selected:
            new_set = selected_set - {k1}
            if not new_set:
                continue
            s1 = sum(Fraction(1, denominators[x]) for x in new_set)
            if s1 < 1:
                d1 = 1 - s1
                if best_deficit is None or d1 < best_deficit:
                    best_deficit = d1
                    best_fixed = new_set
            for k2 in selected:
                if k2 == k1:
                    continue
                new_set2 = selected_set - {k1, k2}
                if not new_set2:
                    continue
                s2 = sum(Fraction(1, denominators[x]) for x in new_set2)
                if s2 < 1:
                    d2 = 1 - s2
                    if best_deficit is None or d2 < best_deficit:
                        best_deficit = d2
                        best_fixed = new_set2
        if best_fixed is not None:
            selected = sorted(best_fixed)
        else:
            sorted_indices = sorted(range(n), key=lambda i: -reciprocals[i])
            sel = []
            cs = Fraction(0)
            for idx in sorted_indices:
                ns = cs + Fraction(1, denominators[idx])
                if ns < 1:
                    sel.append(idx)
                    cs = ns
            selected = sorted(sel) if sel else [0]
    
    # Local search polish
    selected_set = set(selected)
    current_sum = sum(Fraction(1, denominators[k]) for k in selected_set)
    best_deficit = 1 - current_sum
    best_selected = set(selected_set)
    
    for idx in range(n):
        if idx not in selected_set:
            new_sum = current_sum + Fraction(1, denominators[idx])
            if new_sum < 1:
                deficit = 1 - new_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_selected = selected_set | {idx}
                    current_sum = new_sum
                    selected_set = best_selected
    
    for idx in list(selected_set):
        temp_sum = current_sum - Fraction(1, denominators[idx])
        temp_set = selected_set - {idx}
        if not temp_set:
            continue
        for jdx in range(n):
            if jdx not in temp_set:
                new_sum = temp_sum + Fraction(1, denominators[jdx])
                if new_sum < 1:
                    deficit = 1 - new_sum
                    if deficit < best_deficit:
                        best_deficit = deficit
                        best_selected = temp_set | {jdx}
    
    if time.time() > deadline - 2:
        return sorted(best_selected) if best_selected else [0]
    
    sel_list = list(best_selected)
    for i in range(len(sel_list)):
        for j in range(i + 1, len(sel_list)):
            if time.time() > deadline - 3:
                break
            temp_set = set(best_selected)
            temp_set.discard(sel_list[i])
            temp_set.discard(sel_list[j])
            if not temp_set:
                continue
            temp_sum = sum(Fraction(1, denominators[k]) for k in temp_set)
            for a in range(n):
                if a in temp_set:
                    continue
                for b in range(a + 1, n):
                    if b in temp_set:
                        continue
                    new_sum = temp_sum + Fraction(1, denominators[a]) + Fraction(1, denominators[b])
                    if new_sum < 1:
                        deficit = 1 - new_sum
                        if deficit < best_deficit:
                            best_deficit = deficit
                            best_selected = temp_set | {a, b}
    
    return sorted(best_selected) if best_selected else [0]
