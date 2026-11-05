"""Optimized meet-in-the-middle for signed harmonic subsums with multi-phase search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, exp
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
            return False
        a = abs(int_sum)
        if a < best_abs:
            best_abs = a
            best_coeffs = list(coeffs)
            if a == 1:
                return True
        return False
    
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
    def local_search(coeffs, current_sum, max_iters=500):
        improved = True
        iters = 0
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
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
    if update_best(coeffs, current_sum):
        return best_coeffs
    coeffs, current_sum = local_search(coeffs, current_sum)
    if update_best(coeffs, current_sum):
        return best_coeffs
    
    # Simulated Annealing with 2-bit moves
    def simulated_annealing(coeffs, current_sum, max_iters=2000):
        nonlocal best_abs, best_coeffs
        c = list(coeffs)
        s = current_sum
        T = 1000.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random index and change it
            i = random.randint(0, 29)
            old_val = c[i]
            new_val = random.choice([v for v in (-1, 0, 1) if v != old_val])
            c[i] = new_val
            new_s = s - old_val * weights[i] + new_val * weights[i]
            
            if new_s == 0:
                c[i] = old_val
                continue
            
            old_abs = abs(s)
            new_abs = abs(new_s)
            
            if new_abs < old_abs:
                # Accept improvement
                s = new_s
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_coeffs = list(c)
                    if new_abs == 1:
                        return c, s
            else:
                # Accept with probability exp(-delta/T)
                delta = new_abs - old_abs
                if random.random() < exp(-delta / T):
                    s = new_s
            
            T *= cooling
        
        return c, s
    
    # Run simulated annealing
    if time.time() < deadline - 10:
        c, s = simulated_annealing(coeffs, current_sum, max_iters=3000)
        update_best(c, s)
        if best_coeffs is not None and best_abs <= 100:
            # Polish with local search
            c2, s2 = local_search(list(best_coeffs), best_abs * (1 if best_coeffs[0] >= 0 else -1))
            # Recompute sum properly
            s2 = 0
            for i, cv in enumerate(c2):
                if cv != 0:
                    s2 += cv * weights[i]
            update_best(c2, s2)
    
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
    if time.time() > deadline - 15:
        return best_coeffs if best_coeffs else [0] * 28 + [1, -1]
        
    sums1, packs1 = gen_half_packed(w1)
    n1 = len(sums1)
    
    # Create sorted index by sum
    order = sorted(range(n1), key=lambda i: sums1[i])
    sorted_sums = [sums1[i] for i in order]
    
    # Generate half 2 and search
    if time.time() > deadline - 8:
        return best_coeffs if best_coeffs else [0] * 28 + [1, -1]

    sums2, packs2 = gen_half_packed(w2)
    n2 = len(sums2)
    
    # For each entry in half 2, binary search in half 1
    for k in range(n2):
        if time.time() > deadline - 3:
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
                    if a == 1:
                        return best_coeffs
    
    # If MITM found something, do local search to polish
    if best_coeffs is not None:
        coeffs = list(best_coeffs)
        current_sum = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                current_sum += c * weights[i]
        coeffs, current_sum = local_search(coeffs, current_sum, max_iters=100)
        update_best(coeffs, current_sum)
    
    # If still no solution, use fallback
    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
