"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    P = 1
    for d in divisors:
        P = lcm(P, d)
        
    # Iterate window size W
    # We expect W to be small.
    # Upper bound for W: 
    # Worst case, we might need a large window. 
    # But with 10 divisors, and max divisor 39, 
    # a window of size ~100-200 should suffice.
    # Let's cap W at 1000 to be safe, but we expect to find it much sooner.
    
    for W in range(1, 1001):
        # Check if there exists L in [1, P] such that matching exists in [L, L+W]
        # We iterate L. 
        # Optimization: If W is small, many L will be pruned by the "no multiple" check.
        
        for L in range(1, P + 1):
            # Check if each divisor has at least one multiple in [L, L+W]
            valid_L = True
            candidates = [[] for _ in range(n)]
            
            for i, d in enumerate(divisors):
                # First multiple >= L
                # Let L = k*d + r. r = L % d.
                # If r == 0, first multiple is L. dist = 0.
                # If r > 0, first multiple is L + (d - r). dist = d - r.
                r = L % d
                if r == 0:
                    first_m = L
                else:
                    first_m = L + (d - r)
                
                if first_m > L + W:
                    valid_L = False
                    break
                
                # Add multiples
                m = first_m
                while m <= L + W:
                    candidates[i].append(m)
                    m += d
            
            if not valid_L:
                continue
            
            # Bipartite matching
            # owner[num] = i
            owner = {}
            
            def kuhn(i, visited):
                for num in candidates[i]:
                    if num in visited:
                        continue
                    visited.add(num)
                    if num not in owner or kuhn(owner[num], visited):
                        owner[num] = i
                        return True
                return False
            
            matched_count = 0
            for i in range(n):
                visited = set()
                if kuhn(i, visited):
                    matched_count += 1
            
            if matched_count == n:
                # Found a solution
                vals = [0] * n
                for num, div_idx in owner.items():
                    vals[div_idx] = num
                # Verify distinctness (should be guaranteed by matching)
                # Verify bounds
                if min(vals) >= 1 and max(vals) <= max_val:
                    return [int(x) for x in vals]
                    
    # Fallback: This should not be reached if W limit is high enough
    # If reached, we can try a larger W or a different strategy
    # For now, return a valid (possibly suboptimal) solution using a large window
    # Greedy assignment for a large window
    L = 1
    W = 10000
    valid_L = True
    candidates = [[] for _ in range(n)]
    for i, d in enumerate(divisors):
        r = L % d
        if r == 0:
            first_m = L
        else:
            first_m = L + (d - r)
        m = first_m
        while m <= L + W:
            candidates[i].append(m)
            m += d
            
    owner = {}
    def kuhn_fallback(i, visited):
        for num in candidates[i]:
            if num in visited:
                continue
            visited.add(num)
            if num not in owner or kuhn_fallback(owner[num], visited):
                owner[num] = i
                return True
        return False
        
    for i in range(n):
        visited = set()
        kuhn_fallback(i, visited)
        
    vals = [0] * n
    for num, div_idx in owner.items():
        vals[div_idx] = num
        
    return [int(x) for x in vals]
# EVOLVE-BLOCK-END
