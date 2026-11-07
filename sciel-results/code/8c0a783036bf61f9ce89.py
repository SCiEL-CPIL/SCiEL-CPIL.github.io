import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

# Compute LCM of all divisors for periodicity
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)
# LCM_ALL = 30030

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We search for the minimum width W.
    # Lower bound: 10 distinct integers in [L, L+W] implies W >= 9.
    # So we start from W=9.
    
    for W in range(9, 500):
        if time.time() > deadline:
            break
            
        # We scan starting positions L.
        # The pattern of available multiples repeats every LCM_ALL.
        # However, we don't need to scan the full LCM_ALL if we find a solution early.
        # Also, for small W, the number of valid L's might be sparse.
        # But scanning L in [1, LCM_ALL] is safe.
        
        for L in range(1, LCM_ALL + 1):
            if time.time() > deadline:
                break
                
            if L + W > MAX_VAL:
                break
                
            # Check if a valid assignment exists for this L and W
            # Candidates for divisor i are multiples of d_i in [L, L+W]
            candidates = []
            valid = True
            for i in range(N):
                d = DIVISORS[i]
                # First multiple >= L
                first = ((L + d - 1) // d) * d
                cand_list = []
                v = first
                while v <= L + W:
                    cand_list.append(v)
                    v += d
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
            
            # Solve the assignment problem using Kuhn's Algorithm (Bipartite Matching)
            # Left side: Divisors (indices 0..N-1)
            # Right side: Candidate values (unique integers)
            
            # Map values to their indices for quick lookup? 
            # Or just use a dict for match_right: value -> divisor_index
            match_right = {}
            
            # visited set for the current DFS attempt
            # We need to be careful with recursion depth and overhead.
            # N=10, so recursion depth is small.
            
            def try_kuhn(i, visited):
                # i is the divisor index
                for v in candidates[i]:
                    if v in visited:
                        continue
                    visited.add(v)
                    if v not in match_right or try_kuhn(match_right[v], visited):
                        match_right[v] = i
                        return True
                return False
            
            # Try to find a matching for all divisors
            # We can optimize by processing divisors with fewer candidates first?
            # Kuhn's algorithm is order-independent for existence, but performance can vary.
            # Let's stick to 0..N-1 for simplicity, or sort by candidate count for speed.
            # Sorting indices by len(candidates)
            indices = list(range(N))
            indices.sort(key=lambda i: len(candidates[i]))
            
            success = True
            for i in indices:
                visited = set()
                if not try_kuhn(i, visited):
                    success = False
                    break
            
            if success:
                # Extract assignment
                assignment = [0] * N
                for v, i in match_right.items():
                    assignment[i] = v
                
                # Verify just in case (should be valid by construction)
                # Distinctness is guaranteed by matching.
                # Divisibility is guaranteed by candidate generation.
                # Bounds are guaranteed by L, W.
                
                return [int(x) for x in assignment]
    
    # Fallback: return a valid set if no small W found
    # e.g., x_i = d_i * (i+1) * 100
    # This might have a large range, but it's valid.
    return [int(d * (i + 1) * 100) for i, d in enumerate(DIVISORS)]
