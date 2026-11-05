import time
import sys
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 85
    
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
    
    # Extract just the sums for bisect
    B_sums = [x[0] for x in sums_B]
    
    # Find the maximum sum < L using meet-in-the-middle
    best_sum = -1
    best_a_idx = 0
    best_b_idx = 0
    
    for i in range(len(sums_A) - 1, -1, -1):
        a_sum, a_mask = sums_A[i]
        target = L - a_sum
        
        # Find the largest j where B_sums[j] < target
        pos = bisect.bisect_left(B_sums, target)
        
        if pos > 0:
            candidate_j = pos - 1
            b_sum = B_sums[candidate_j]
            total = a_sum + b_sum
            if total > best_sum:
                best_sum = total
                best_a_idx = i
                best_b_idx = candidate_j
                
        # Time check
        if time.time() > deadline - 20:
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
            
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac >= 1:
        return [0, 1, 5]
    
    # Local search polish: try swapping elements to improve
    current_sum = best_sum
    current_set = selected
    
    # Try to improve by single swaps
    for _ in range(50):  # Limit iterations
        if time.time() > deadline - 10:
            break
            
        improved = False
        # Try removing one element
        for elem in list(current_set):
            new_sum = current_sum - weights[elem]
            if 0 <= new_sum < L:
                # Check if removing improves (we want to maximize sum < L)
                # Actually, removing decreases sum, so it might allow adding more
                pass
        
        # Try adding one element not in set
        for elem in range(n):
            if elem not in current_set:
                new_sum = current_sum + weights[elem]
                if 0 <= new_sum < L and new_sum > current_sum:
                    current_set.add(elem)
                    current_sum = new_sum
                    improved = True
                    break
        
        # Try single swap: remove one, add one
        if not improved:
            for remove_elem in list(current_set):
                base_sum = current_sum - weights[remove_elem]
                for add_elem in range(n):
                    if add_elem not in current_set:
                        new_sum = base_sum + weights[add_elem]
                        if 0 <= new_sum < L and new_sum > current_sum:
                            current_set.remove(remove_elem)
                            current_set.add(add_elem)
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    # Convert to sorted list
    result = sorted(current_set)
    
    # Final verification
    total_frac = sum(Fraction(1, denominators[k]) for k in result)
    if total_frac < 1:
        return result
    else:
        return sorted(selected)
