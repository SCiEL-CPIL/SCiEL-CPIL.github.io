import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 30
    L = reduce(math.lcm, range(1, n + 1))
    W = [L // i for i in range(1, n + 1)]
    
    # Strategy: Use meet-in-the-middle with modular arithmetic
    # Split into 3 parts to reduce memory
    # Part 1: terms 1-10, Part 2: terms 11-20, Part 3: terms 21-30
    
    # First, let's try a greedy + local search approach which is much faster
    coeffs = _greedy_initial(W, n)
    best_val = _evaluate(coeffs, W)
    
    if best_val == 1:
        return coeffs
    
    # Try local search / hill climbing
    improved = _local_search(coeffs, W, best_val, deadline)
    if improved is not None:
        return improved
    
    # Try meet-in-the-middle with smaller groups
    # Group 1: first 10 terms
    # Group 2: middle 10 terms  
    # Group 3: last 10 terms
    
    mid1 = 10
    mid2 = 20
    
    W1 = W[:mid1]
    W2 = W[mid1:mid2]
    W3 = W[mid2:]
    
    # Generate all sums for group 1 (3^10 = 59049, manageable)
    sums1 = _generate_sums(W1)
    
    # Generate all sums for group 2 (3^10 = 59049, manageable)
    sums2 = _generate_sums(W2)
    
    # For group 3, we'll use a different strategy
    # We want to find s1 + s2 + s3 = k for small nonzero k
    # So s3 = k - s1 - s2
    
    # Precompute sums3 as a set
    sums3 = _generate_sums(W3)
    set3 = set(sums3)
    
    best_total = best_val
    best_coeffs = coeffs
    
    # Try small values of k
    for k_abs in range(1, 50):
        for k in [k_abs, -k_abs]:
            if time.time() > deadline:
                break
                
            # We need s1 + s2 + s3 = k
            # So s3 = k - s1 - s2
            # For each s1, s2, check if k - s1 - s2 is in set3
            
            found = False
            for s1 in sums1:
                if time.time() > deadline:
                    break
                for s2 in sums2:
                    s3_needed = k - s1 - s2
                    if s3_needed in set3:
                        # Found a solution!
                        c1 = _find_coeffs(s1, W1)
                        c2 = _find_coeffs(s2, W2)
                        c3 = _find_coeffs(s3_needed, W3)
                        
                        if c1 is not None and c2 is not None and c3 is not None:
                            final_coeffs = c1 + c2 + c3
                            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
                            if total == k and abs(k) < best_total:
                                best_total = abs(k)
                                best_coeffs = final_coeffs
                                found = True
                                break
                if found:
                    break
            if found:
                break
        if best_total == 1:
            break
    
    return best_coeffs

def _generate_sums(W):
    """Generate all possible sums for given weights using iterative approach."""
    sums = [0]
    for w in W:
        new_sums = []
        for s in sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        sums = new_sums
    return sums

def _find_coeffs(target, W):
    """Recursively find coefficients for the given weights to sum to target."""
    if not W:
        return [] if target == 0 else None
        
    w = W[0]
    rest_W = W[1:]
    
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, rest_W)
        if res is not None:
            return [c] + res
    return None

def _evaluate(coeffs, W):
    """Evaluate the absolute value of the signed sum."""
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    return abs(total) if total != 0 else float('inf')

def _greedy_initial(W, n):
    """Create an initial solution using greedy approach."""
    coeffs = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        w = W[i]
        best_c = 0
        best_val = curr
        for c in [-1, 0, 1]:
            val = curr + c * w
            if abs(val) < abs(best_val) or (abs(val) == abs(best_val) and val != 0):
                best_val = val
                best_c = c
        coeffs[i] = best_c
        curr = best_val
    
    if curr == 0:
        # Make nonzero by setting smallest weight to 1
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
    
    return coeffs

def _local_search(coeffs, W, current_best, deadline):
    """Try local search to improve the solution."""
    n = len(coeffs)
    coeffs = list(coeffs)
    best_val = current_best
    
    # Try single changes
    for i in range(n):
        if time.time() > deadline:
            break
        orig = coeffs[i]
        for c in [-1, 0, 1]:
            if c == orig:
                continue
            coeffs[i] = c
            val = _evaluate(coeffs, W)
            if val < best_val:
                best_val = val
                if best_val == 1:
                    return coeffs
            else:
                coeffs[i] = orig
    
    # Try double changes
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            orig_i = coeffs[i]
            orig_j = coeffs[j]
            for ci in [-1, 0, 1]:
                for cj in [-1, 0, 1]:
                    coeffs[i] = ci
                    coeffs[j] = cj
                    val = _evaluate(coeffs, W)
                    if val < best_val:
                        best_val = val
                        if best_val == 1:
                            return coeffs
                    else:
                        coeffs[i] = orig_i
                        coeffs[j] = orig_j
    
    if best_val < current_best:
        return coeffs
    return None
