import time
import math
from math import gcd
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 30
    
    # Compute LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Scaled weights: w[i] = L // (i+1) for i in 0..29 (corresponding to 1/i)
    weights = [L // (i + 1) for i in range(n)]
    
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
    
    # Greedy initialization: process largest weights first
    def greedy_init():
        coeffs = [0] * n
        current_sum = 0
        # Sort indices by weight descending
        indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
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
        
        # Ensure nonzero
        if current_sum == 0:
            for i in range(n):
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
    
    # Local search: try changing one coefficient at a time
    def local_search(coeffs, current_sum, max_iter=10000):
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 3:
                break
            for i in range(n):
                if time.time() > deadline - 3:
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
    
    # Simulated annealing for better local search
    def simulated_annealing(coeffs, current_sum, max_iter=50000):
        import random
        temp = 1.0
        cooling = 0.9995
        best_local = abs(current_sum)
        best_local_coeffs = list(coeffs)
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            i = random.randint(0, n - 1)
            old_val = coeffs[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            coeffs[i] = new_val
            new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
            if new_sum == 0:
                coeffs[i] = old_val
                continue
            delta = abs(new_sum) - abs(current_sum)
            if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-10)):
                current_sum = new_sum
                if abs(current_sum) < best_local:
                    best_local = abs(current_sum)
                    best_local_coeffs = list(coeffs)
                    if best_local == 1:
                        break
            else:
                coeffs[i] = old_val
            temp *= cooling
        
        return best_local_coeffs, current_sum if abs(current_sum) == best_local else 0
    
    # Get initial solution via greedy
    coeffs, current_sum = greedy_init()
    if update_best(coeffs, current_sum):
        return best_coeffs
    
    # Polish with local search
    coeffs, current_sum = local_search(coeffs, current_sum)
    if update_best(coeffs, current_sum):
        return best_coeffs
    
    # Try simulated annealing from this point
    if time.time() < deadline - 10:
        sa_coeffs, sa_sum = simulated_annealing(coeffs, current_sum)
        if sa_sum != 0 and abs(sa_sum) < best_abs:
            best_abs = abs(sa_sum)
            best_coeffs = sa_coeffs
            if best_abs == 1:
                return best_coeffs
    
    # Meet-in-the-middle with bit-packed coefficients
    half = 15
    idx1 = list(range(half))
    idx2 = list(range(half, n))
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    def gen_half_packed(w_list):
        """Generate all (sum, packed_coeffs) for 15 indices."""
        n_w = len(w_list)
        sums_arr = [0]
        packs_arr = [0]
        for j in range(n_w):
            w = w_list[j]
            new_sums = []
            new_packs = []
            for k in range(len(sums_arr)):
                s = sums_arr[k]
                p = packs_arr[k]
                # val = -1: bits 2j,2j+1 = 10 (binary)
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
    
    # Generate first half
    if time.time() > deadline - 20:
        return best_coeffs if best_coeffs else [0] * 28 + [1, -1]
    
    sums1, packs1 = gen_half_packed(w1)
    n1 = len(sums1)
    
    # Sort first half by sum
    order = sorted(range(n1), key=lambda i: sums1[i])
    sorted_sums = [sums1[i] for i in order]
    
    # Generate second half and search
    if time.time() > deadline - 10:
        return best_coeffs if best_coeffs else [0] * 28 + [1, -1]
    
    sums2, packs2 = gen_half_packed(w2)
    n2 = len(sums2)
    
    # For each entry in half 2, binary search in half 1
    found_improvement = False
    for k in range(n2):
        if time.time() > deadline - 3:
            break
        s2 = sums2[k]
        target = -s2
        pos = bisect_left(sorted_sums, target)
        
        # Check positions around the target
        for p in range(max(0, pos - 5), min(n1, pos + 6)):
            total = sorted_sums[p] + s2
            if total == 0:
                continue
            a = abs(total)
            if a < best_abs:
                # Reconstruct coefficients
                idx_in_order = order[p]
                p1 = packs1[idx_in_order]
                p2 = packs2[k]
                new_coeffs = [0] * n
                for j in range(half):
                    b1 = (p1 >> (2*j)) & 3
                    if b1 == 1:
                        new_coeffs[idx1[j]] = 1
                    elif b1 == 2:
                        new_coeffs[idx1[j]] = -1
                    b2 = (p2 >> (2*j)) & 3
                    if b2 == 1:
                        new_coeffs[idx2[j]] = 1
                    elif b2 == 2:
                        new_coeffs[idx2[j]] = -1
                best_abs = a
                best_coeffs = new_coeffs
                found_improvement = True
                if a == 1:
                    return best_coeffs
    
    # If MITM found something, polish with local search
    if best_coeffs is not None and found_improvement:
        coeffs = list(best_coeffs)
        current_sum = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                current_sum += c * weights[i]
        coeffs, current_sum = local_search(coeffs, current_sum)
        if current_sum != 0 and abs(current_sum) < best_abs:
            best_abs = abs(current_sum)
            best_coeffs = coeffs
            if best_abs == 1:
                return best_coeffs
    
    # If still no good solution, use fallback
    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs
