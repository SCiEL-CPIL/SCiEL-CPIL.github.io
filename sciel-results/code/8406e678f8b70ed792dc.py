"""Optimized meet-in-the-middle for signed harmonic subsums."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    deadline = time.time() + 82  # 90s limit, 8s margin
    
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
    
    # Local search
    def local_search(coeffs, current_sum):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(30):
                if time.time() > deadline - 2:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    break
        return coeffs, current_sum
    
    # Get initial solution
    coeffs, current_sum = greedy_init()
    update_best(coeffs, current_sum)
    coeffs, current_sum = local_search(coeffs, current_sum)
    update_best(coeffs, current_sum)
    
    # Meet-in-the-middle with bit-packed coefficients
    half = 15
    idx1 = list(range(half))
    idx2 = list(range(half, 30))
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    def gen_half_packed(w_list):
        """Generate all (sum, packed_coeffs) for 15 indices using array-based generation."""
        n = len(w_list)
        # Start with one entry: sum=0, packed=0
        sums_arr = [0]
        packs_arr = [0]
        for j in range(n):
            w = w_list[j]
            # For each existing entry, create 3 new ones
            # We need to be careful about memory
            new_sums = []
            new_packs = []
            for k in range(len(sums_arr)):
                s = sums_arr[k]
                p = packs_arr[k]
                # val = -1: bits 2j,2j+1 = 10
                new_sums.append(s - w)
                new_packs.append(p | (2 << (2*j)))
                # val = 0: bits 2j,2j+1 = 00
                new_sums.append(s)
                new_packs.append(p)
                # val = 1: bits 2j,2j+1 = 01
                new_sums.append(s + w)
                new_packs.append(p | (1 << (2*j)))
            sums_arr = new_sums
            packs_arr = new_packs
        return sums_arr, packs_arr
    
    # Generate half 1
    t_start = time.time()
    sums1, packs1 = gen_half_packed(w1)
    t_gen1 = time.time() - t_start
    n1 = len(sums1)
    
    # Create sorted index by sum
    order = sorted(range(n1), key=lambda i: sums1[i])
    sorted_sums = [sums1[i] for i in order]
    
    # Generate half 2 and search
    t_start = time.time()
    sums2, packs2 = gen_half_packed(w2)
    t_gen2 = time.time() - t_start
    n2 = len(sums2)
    
    # For each entry in half 2, binary search in half 1
    found_better = False
    for k in range(n2):
        if time.time() > deadline - 5:
            break
        s2 = sums2[k]
        target = -s2
        # Binary search for target in sorted_sums
        pos = bisect_left(sorted_sums, target)
        # Check pos and pos-1
        for p in (pos, pos - 1):
            if 0 <= p < n1:
                total = sorted_sums[p] + s2
                if total == 0:
                    continue
                a = abs(total)
                if a < best_abs:
                    # Reconstruct coefficients
                    idx_in_order = order[p]
                    p1 = packs1[idx_in_order]
                    p2 = packs2[k]
                    coeffs = [0] * 30
                    for j in range(half):
                        b1 = (p1 >> (2*j)) & 3
                        if b1 == 1:
                            coeffs[idx1[j]] = 1
                        elif b1 == 2:
                            coeffs[idx1[j]] = -1
                        b2 = (p2 >> (2*j)) & 3
                        if b2 == 1:
                            coeffs[idx2[j]] = 1
                        elif b2 == 2:
                            coeffs[idx2[j]] = -1
                    best_abs = a
                    best_coeffs = coeffs
                    found_better = True
    
    # If MITM found something, do local search to polish
    if best_coeffs is not None:
        coeffs = list(best_coeffs)
        current_sum = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                current_sum += c * weights[i]
        coeffs, current_sum = local_search(coeffs, current_sum)
        update_best(coeffs, current_sum)
    
    # If still no solution, use fallback
    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
