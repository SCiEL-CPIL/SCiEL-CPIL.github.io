import time
import random
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    best_val = float('inf')
    best_assign = None
    
    # LCM of all divisors
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    # For a given window [L, L+W], find all multiples of divisor d in that window
    def get_multiples_in_window(d, L, W):
        # smallest multiple >= L
        k_min = (L + d - 1) // d
        # largest multiple <= L+W
        k_max = (L + W) // d
        if k_min > k_max:
            return []
        return list(range(k_min * d, k_max * d + 1, d))
    
    # Backtracking to assign distinct multiples
    def backtrack(idx, used, assignment):
        nonlocal best_val, best_assign
        if idx == n:
            val = max(assignment) - min(assignment)
            if val < best_val:
                best_val = val
                best_assign = assignment[:]
            return True
        
        d = divisors[idx]
        candidates = get_multiples_in_window(d, L, W)
        # Shuffle for diversity in random restarts
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in used:
                used.add(c)
                assignment[idx] = c
                backtrack(idx + 1, used, assignment)
                used.remove(c)
        
        return False
    
    # Try increasing window sizes
    for W in range(1, 300):
        if time.time() > deadline:
            break
        
        # Try several random L positions
        num_tries = 50 if W < 50 else 20
        for trial in range(num_tries):
            if time.time() > deadline:
                break
            
            L = random.randint(1, MAX_V - W)
            assignment = [0] * n
            used = set()
            if backtrack(0, used, assignment):
                if best_val <= W:
                    # We found something within this window size
                    break
        
        # If we found a solution with value <= W, we can stop
        # since we're trying W in increasing order
        if best_val <= W:
            break
    
    # Fallback: if no solution found (shouldn't happen), use the divisors themselves
    if best_assign is None:
        best_assign = divisors[:]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_assign]
