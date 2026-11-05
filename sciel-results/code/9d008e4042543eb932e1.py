import time
import math
import bisect
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    reciprocals = [1.0 / d for d in denominators]
    
    mid = n // 2
    recips_A = reciprocals[:mid]
    recips_B = reciprocals[mid:]
    
    def gen_sums_fp(recips_subset):
        sums = [0.0]
        masks = [0]
        for i, r in enumerate(recips_subset):
            current_len = len(sums)
            for k in range(current_len):
                s = sums[k]
                m = masks[k]
                ns = s + r
                nm = m | (1 << i)
                if ns < 1.0:
                    sums.append(ns)
                    masks.append(nm)
        return list(zip(sums, masks))
    
    sums_A = gen_sums_fp(recips_A)
    sums_B = gen_sums_fp(recips_B)
    
    sums_B.sort(key=lambda x: x[0])
    sums_B_values = [s for s, m in sums_B]
    
    best_deficit = 1.0
    best_mask_A = 0
    best_mask_B = 0
    
    for s_A, m_A in sums_A:
        target = 1.0 - s_A
        idx = bisect.bisect_left(sums_B_values, target)
        
        if idx > 0:
            s_B, m_B = sums_B[idx - 1]
            total_sum = s_A + s_B
            deficit = 1.0 - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask_A = m_A
                best_mask_B = m_B
                
                if best_deficit < 1e-15:
                    break
    
    selected_indices = []
    for i in range(len(recips_A)):
        if best_mask_A & (1 << i):
            selected_indices.append(i)
    for i in range(len(recips_B)):
        if best_mask_B & (1 << i):
            selected_indices.append(mid + i)
    
    if not selected_indices:
        return [0]
    
    # Local search with exact fractions
    frac_vals = [Fraction(1, d) for d in denominators]
    best_set = set(selected_indices)
    current_sum = sum(frac_vals[i] for i in best_set)
    best_deficit_frac = Fraction(1, 1) - current_sum
    
    if time.time() - start < deadline - 5:
        improved = True
        while improved and (time.time() - start) < deadline - 5:
            improved = False
            
            # Try adding
            for i in range(n):
                if i not in best_set:
                    new_sum = current_sum + frac_vals[i]
                    new_deficit = Fraction(1, 1) - new_sum
                    if 0 < new_deficit < best_deficit_frac:
                        best_deficit_frac = new_deficit
                        current_sum = new_sum
                        best_set.add(i)
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try removing
            for i in list(best_set):
                new_sum = current_sum - frac_vals[i]
                new_deficit = Fraction(1, 1) - new_sum
                if 0 < new_deficit < best_deficit_frac:
                    best_deficit_frac = new_deficit
                    current_sum = new_sum
                    best_set.remove(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping
            for i in list(best_set):
                for j in range(n):
                    if j not in best_set:
                        new_sum = current_sum - frac_vals[i] + frac_vals[j]
                        new_deficit = Fraction(1, 1) - new_sum
                        if 0 < new_deficit < best_deficit_frac:
                            best_deficit_frac = new_deficit
                            current_sum = new_sum
                            best_set.remove(i)
                            best_set.add(j)
                            improved = True
                            break
                if improved:
                    break
    
    if best_set:
        selected_indices = sorted(best_set)
    else:
        selected_indices = [0]
    
    return selected_indices

if __name__ == "__main__":
    res = construct()
    print(res)
