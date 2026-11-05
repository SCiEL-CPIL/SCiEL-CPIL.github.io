"""Optimized meet-in-the-middle for signed harmonic subsums."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    # Best solution tracking
    best_abs = float('inf')
    best_coeffs = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_abs, best_coeffs
        if int_sum == 0:
            return
        a = abs(int_sum)
        if a < best_abs:
            best_abs = a
            best_coeffs = list(coeffs)
    
    # Greedy initialization
    def greedy_init():
        coeffs = [0] * 30
        current_sum = 0
        indices = sorted(range(30), key=lambda i: weights[i], reverse=True)
        for i in indices:
            best_val = 0
            best_abs_local = abs(current_sum) if current_sum != 0 else float('inf')
            for val in (-1, 0, 1):
                new_sum = current_sum + val * weights[i]
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < best_abs_local:
                    best_abs_local = new_abs
                    best_val = val
            coeffs[i] = best_val
            current_sum += best_val * weights[i]
        if current_sum == 0:
            for i in range(30):
                old = coeffs[i]
                for new in (-1, 1):
                    coeffs[i] = new
                    new_sum = current_sum - old * weights[i] + new * weights[i]
                    if new_sum != 0:
                        current_sum = new_sum
                        return coeffs, current_sum
                coeffs[i] = old
            coeffs[0] = 1
            current_sum = weights[0]
        return coeffs, current_sum
    
    # Local search with 1-flip and 2-flip moves
    def local_search(coeffs, current_sum):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 1-flip moves
            for i in range(30):
                if time.time() > deadline - 5:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                    if new_sum == 0:
                        continue
                    if abs(new_sum) < abs(current_sum):
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            
            # 2-flip moves if no 1-flip improvement
            if not improved:
                for i in range(30):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, 30):
                        if time.time() > deadline - 5:
                            break
                        old_i = coeffs[i]
                        old_j = coeffs[j]
                        for new_i in (-1, 0, 1):
                            if new_i == old_i:
                                continue
                            for new_j in (-1, 0, 1):
                                if new_j == old_j:
                                    continue
                                new_sum = current_sum - old_i * weights[i] - old_j * weights[j] + new_i * weights[i] + new_j * weights[j]
                                if new_sum == 0:
                                    continue
                                if abs(new_sum) < abs(current_sum):
                                    coeffs[i] = new_i
                                    coeffs[j] = new_j
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return coeffs, current_sum
    
    # Get initial solution
    coeffs, current_sum = greedy_init()
    update_best(coeffs, current_sum)
    coeffs, current_sum = local_search(coeffs, current_sum)
    update_best(coeffs, current_sum)
    
    # Meet-in-the-middle
    half = 15
    idx1 = list(range(half))
    idx2 = list(range(half, 30))
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    def gen_half_packed(w_list):
        """Generate all (sum, packed_coeffs) for 15 indices."""
        n = len(w_list)
        sums_arr = [0]
        packs_arr = [0]
        for j in range(n):
            w = w_list[j]
            new_sums = []
            new_packs = []
            for k in range(len(sums_arr)):
                s = sums_arr[k]
                p = packs_arr[k]
                # val = -1
                new_sums.append(s - w)
                new_packs.append(p | (2 << (2*j)))
                # val = 0
                new_sums.append(s)
                new_packs.append(p)
                # val = 1
                new_sums.append(s + w)
                new_packs.append(p | (1 << (2*j)))
            sums_arr = new_sums
            packs_arr = new_packs
        return sums_arr, packs_arr
    
    # Generate half 1
    t_start = time.time()
    sums1, packs1 = gen_half_packed(w1)
    n1 = len(sums1)
    
    # Sort half 1
    # We need to sort sums1 and keep packs1 aligned
    # Using zip and sorted is slow for 14M items.
    # Let's use a list of tuples? No, memory.
    # Let's sort indices.
    order1 = sorted(range(n1), key=sums1.__getitem__)
    sorted_sums1 = [sums1[i] for i in order1]
    sorted_packs1 = [packs1[i] for i in order1]
    
    # Generate half 2
    sums2, packs2 = gen_half_packed(w2)
    n2 = len(sums2)
    
    # Prepare half 2 for linear scan: we want -sums2
    # Let A = sorted_sums1
    # Let B = sorted(-sums2)
    # We want min |A[i] - B[j]|
    neg_sums2 = [-s for s in sums2]
    order2 = sorted(range(n2), key=neg_sums2.__getitem__)
    sorted_neg_sums2 = [neg_sums2[i] for i in order2]
    sorted_packs2 = [packs2[i] for i in order2]
    
    # Linear scan
    i = 0
    j = 0
    best_diff = float('inf')
    best_i = -1
    best_j = -1
    
    while i < n1 and j < n2:
        if time.time() > deadline - 10:
            break
        a_val = sorted_sums1[i]
        b_val = sorted_neg_sums2[j]
        diff = a_val - b_val
        abs_diff = abs(diff)
        
        if diff != 0 and abs_diff < best_diff:
            best_diff = abs_diff
            best_i = i
            best_j = j
            
        if a_val < b_val:
            i += 1
        else:
            j += 1
            
    if best_i != -1:
        # Reconstruct coefficients
        p1 = sorted_packs1[best_i]
        p2 = sorted_packs2[best_j]
        coeffs = [0] * 30
        for k in range(half):
            b1 = (p1 >> (2*k)) & 3
            if b1 == 1:
                coeffs[idx1[k]] = 1
            elif b1 == 2:
                coeffs[idx1[k]] = -1
            b2 = (p2 >> (2*k)) & 3
            if b2 == 1:
                coeffs[idx2[k]] = 1
            elif b2 == 2:
                coeffs[idx2[k]] = -1
                
        # Calculate sum to verify and update
        current_sum = 0
        for k in range(30):
            if coeffs[k] != 0:
                current_sum += coeffs[k] * weights[k]
                
        update_best(coeffs, current_sum)
        
        # Polish with local search
        coeffs, current_sum = local_search(coeffs, current_sum)
        update_best(coeffs, current_sum)
        
    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
