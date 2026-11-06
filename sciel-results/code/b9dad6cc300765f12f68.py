import time
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 90s limit (evaluator kills at 90s)
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # LCM of 2..48. This is a fixed constant.
    # LCM(1..48) = 5342931457063200
    # We use this to convert 1/d into integer weights: w_d = LCM // d
    # Sum of weights < LCM  <=>  Sum of reciprocals < 1
    # This allows fast integer comparison without Fraction overhead in the inner loop.
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index 0 -> denom 2, weight = L // 2
    # index i -> denom i+2, weight = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending (which is same as denominator ascending)
    # Since we want to maximize sum, we want to pick large weights (small denominators) first.
    # The input order 0..46 is already sorted by weight descending.
    # So we can just use the range(0, n) directly, but let's be explicit.
    # sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    # Actually, 1/2 > 1/3 > ... so index 0 has largest weight.
    
    # We will search for a subset S such that sum(weights[i] for i in S) < L.
    # Maximize sum(weights[i]).
    
    best_sum_weight = 0
    best_indices = []
    
    # Helper to update best
    def update_best(sum_w, indices):
        nonlocal best_sum_weight, best_indices
        if sum_w > best_sum_weight:
            best_sum_weight = sum_w
            best_indices = list(indices)
    
    # Strategy:
    # The optimal solution is likely composed of a few "large" fractions (small denominators) 
    # and a "tail" of smaller fractions greedily added.
    # 
    # We will enumerate all subsets of size k=1, 2, 3, 4 as "anchors".
    # For each anchor, we greedily add the remaining largest weights that fit.
    # 
    # Why this works better than DFS:
    # 1. The search space of anchors is small: C(47, 1) + C(47, 2) + C(47, 3) + C(47, 4) ~ 130k.
    # 2. The greedy fill is O(n) per anchor.
    # 3. Total operations ~ 130k * 47 ~ 6 million, which is fast in Python.
    # 4. It avoids the exponential blowup of DFS while exploring a wide variety of "starting points".
    # 5. It uses integer arithmetic for the core loop, which is much faster than Fraction.
    
    # Precompute prefix sums of weights? No, we need to skip anchor elements.
    # We can just iterate through all items and check if they are in the anchor.
    
    # To speed up the "greedy fill", we can pre-sort the items by weight descending.
    # The items are already sorted by weight descending in `weights`.
    # So we just iterate i from 0 to n-1.
    
    # Optimization: 
    # If the anchor sum is already close to L, the greedy fill will add very few items.
    # If the anchor sum is small, the greedy fill will add many.
    
    # Let's implement the enumeration.
    
    # k=1
    for i in range(n):
        anchor_sum = weights[i]
        if anchor_sum >= L: continue
        # Greedy fill
        current_sum = anchor_sum
        current_indices = [i]
        for j in range(n):
            if j == i: continue
            if current_sum + weights[j] < L:
                current_sum += weights[j]
                current_indices.append(j)
        update_best(current_sum, current_indices)
        
        if time.time() - start_time > timeout:
            break
            
    if time.time() - start_time > timeout:
        return best_indices

    # k=2
    for i in range(n):
        for j in range(i + 1, n):
            anchor_sum = weights[i] + weights[j]
            if anchor_sum >= L: continue
            # Greedy fill
            current_sum = anchor_sum
            current_indices = [i, j]
            for k in range(n):
                if k == i or k == j: continue
                if current_sum + weights[k] < L:
                    current_sum += weights[k]
                    current_indices.append(k)
            update_best(current_sum, current_indices)
            
            if time.time() - start_time > timeout:
                return best_indices

    if time.time() - start_time > timeout:
        return best_indices

    # k=3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                anchor_sum = weights[i] + weights[j] + weights[k]
                if anchor_sum >= L: continue
                # Greedy fill
                current_sum = anchor_sum
                current_indices = [i, j, k]
                for m in range(n):
                    if m == i or m == j or m == k: continue
                    if current_sum + weights[m] < L:
                        current_sum += weights[m]
                        current_indices.append(m)
                update_best(current_sum, current_indices)
                
                if time.time() - start_time > timeout:
                    return best_indices

    if time.time() - start_time > timeout:
        return best_indices

    # k=4
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    anchor_sum = weights[i] + weights[j] + weights[k] + weights[m]
                    if anchor_sum >= L: continue
                    # Greedy fill
                    current_sum = anchor_sum
                    current_indices = [i, j, k, m]
                    for p in range(n):
                        if p == i or p == j or p == k or p == m: continue
                        if current_sum + weights[p] < L:
                            current_sum += weights[p]
                            current_indices.append(p)
                    update_best(current_sum, current_indices)
                    
                    if time.time() - start_time > timeout:
                        return best_indices

    if time.time() - start_time > timeout:
        return best_indices

    # If we have time, we can do a local search on the best solution found.
    # Try to swap elements in the best solution to improve the sum.
    # But the greedy fill is already quite good.
    # A more effective local search: 
    # Take the best set. Try to remove one element and re-greedy fill?
    # Or try to add one element if it fits? (Greedy already did this).
    # The greedy fill is optimal for the fixed anchor.
    # The only way to improve is to find a better anchor.
    # We've exhausted k=1..4.
    
    # What about k=5? 
    # C(47, 5) is ~ 1.5 million. 1.5M * 47 ~ 70M operations.
    # This might be too slow in Python (approx 5-10 seconds).
    # Let's try it if we have time.
    
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 5.0:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for m in range(k + 1, n):
                        for p in range(m + 1, n):
                            anchor_sum = weights[i] + weights[j] + weights[k] + weights[m] + weights[p]
                            if anchor_sum >= L: continue
                            # Greedy fill
                            current_sum = anchor_sum
                            current_indices = [i, j, k, m, p]
                            for q in range(n):
                                if q in (i, j, k, m, p): continue
                                if current_sum + weights[q] < L:
                                    current_sum += weights[q]
                                    current_indices.append(q)
                            update_best(current_sum, current_indices)
                            
                            if time.time() - start_time > timeout:
                                return best_indices
    else:
        # If time is tight, skip k=5
        pass

    # Final verification and return
    # The task requires returning zero-based indices.
    # best_indices contains indices into the `denominators` list, which is 0..46 corresponding to 2..48.
    # This matches the requirement.
    
    if not best_indices:
        return [0] # Fallback: 1/2
        
    return best_indices
