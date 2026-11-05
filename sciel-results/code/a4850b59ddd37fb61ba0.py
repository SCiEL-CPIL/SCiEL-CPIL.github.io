import time
import sys
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 80  # 80 second budget to be safe within 90s
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # Split into two sets A and B
    mid = n // 2
    A_weights = weights[:mid]
    B_weights = weights[mid:]
    
    # Generate subset sums for A
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        current_len = len(sums_A)
        for k in range(current_len):
            s, m = sums_A[k]
            sums_A.append((s + w, m | (1 << i)))
            
    # Sort A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Generate subset sums for B
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            sums_B.append((s + w, m | (1 << i)))
            
    # Sort B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # Two-pointer technique to find max A_sum + B_sum < L
    best_sum = -1
    best_a_idx = 0
    best_b_idx = 0
    
    # Create a separate list of B sums for faster bisect
    B_sums = [x[0] for x in sums_B]
    
    # Iterate A from largest to smallest
    for i in range(len(sums_A) - 1, -1, -1):
        a_sum, a_mask = sums_A[i]
        
        # Find the largest j such that a_sum + sums_B[j][0] < L
        target = L - a_sum
        pos = bisect.bisect_left(B_sums, target)
        
        if pos > 0:
            candidate_j = pos - 1
            b_sum, b_mask = sums_B[candidate_j]
            total = a_sum + b_sum
            if total > best_sum:
                best_sum = total
                best_a_idx = i
                best_b_idx = candidate_j
                
        # Time check
        if time.time() > deadline - 30:
            break

    if best_sum < 0:
        return [0, 1, 5]

    # Reconstruct indices
    a_mask = sums_A[best_a_idx][1]
    b_mask = sums_B[best_b_idx][1]
    
    selected = set()
    for i in range(mid):
        if a_mask & (1 << i):
            selected.add(i)
    for i in range(n - mid):
        if b_mask & (1 << i):
            selected.add(mid + i)
            
    # Local search polish: try to improve by swapping elements
    # Current deficit in integer terms
    current_sum = sum(weights[i] for i in selected)
    current_deficit = L - current_sum
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap = None
        best_swap_deficit = current_deficit
        
        # Try adding one element and removing another (or just adding/removing)
        for add_idx in range(n):
            if add_idx in selected:
                continue
            add_w = weights[add_idx]
            
            # Try just adding
            new_sum = current_sum + add_w
            if new_sum < L:
                new_deficit = L - new_sum
                if new_deficit < best_swap_deficit:
                    best_swap_deficit = new_deficit
                    best_swap = (add_idx, None)
            
            # Try swapping: add this, remove one from selected
            for rem_idx in list(selected):
                rem_w = weights[rem_idx]
                new_sum2 = current_sum + add_w - rem_w
                if new_sum2 < L:
                    new_deficit2 = L - new_sum2
                    if new_deficit2 < best_swap_deficit:
                        best_swap_deficit = new_deficit2
                        best_swap = (add_idx, rem_idx)
        
        if best_swap is not None:
            add_idx, rem_idx = best_swap
            selected.add(add_idx)
            if rem_idx is not None:
                selected.discard(rem_idx)
            current_sum = sum(weights[i] for i in selected)
            current_deficit = L - current_sum
            improved = True
    
    # Convert back to list
    result = sorted(selected)
    
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in result)
    if total_frac < 1:
        return result
    else:
        return [0, 1, 5]
