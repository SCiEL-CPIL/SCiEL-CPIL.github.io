import math
import time
from functools import lru_cache

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030  # LCM of divisors
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    def find_sdr(sets):
        """
        Finds a System of Distinct Representatives.
        Returns a list of chosen values or None if impossible.
        """
        # Sort indices by domain size (MRV heuristic)
        indices = list(range(len(sets)))
        indices.sort(key=lambda i: len(sets[i]))
        
        if any(len(sets[i]) == 0 for i in indices):
            return None
            
        used = set()
        assignment = [None] * len(sets)
        
        def dfs(idx):
            if idx == len(sets):
                return True
            i = indices[idx]
            # Optional: Forward Checking
            # Check if any other unassigned set has no available candidates
            # For now, simple backtracking
            for val in sets[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if dfs(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if dfs(0):
            return assignment
        return None

    def exists_sdr_for_W(W):
        """
        Checks if there is any start S such that an SDR exists in window [S, S+W].
        Returns the assignment and S if found, else None.
        """
        for S in range(L):
            if time.time() > deadline:
                return None
                
            current_sets = []
            valid = True
            for i in range(n):
                d = divisors[i]
                # Find candidates in [S, S+W] modulo L
                candidates = []
                
                # Case 1: p in [S, min(S+W, L-1)]
                p = ((S + d - 1) // d) * d
                limit1 = min(S + W, L - 1)
                while p <= limit1:
                    candidates.append(p)
                    p += d
                    
                # Case 2: wrap around. If S + W >= L, [0, S+W-L] are valid.
                if S + W >= L:
                    limit2 = S + W - L
                    p2 = 0
                    while p2 <= limit2:
                        candidates.append(p2)
                        p2 += d
                        
            if not candidates:
                valid = False
                break
                
            current_sets.append(candidates)
            
            if not valid:
                continue
                
            # Check SDR
            assignment = find_sdr(current_sets)
            if assignment is not None:
                # Construct values
                vals = []
                for i in range(n):
                    p = assignment[i]
                    if p >= S:
                        x = p
                    else:
                        x = p + L
                    if x <= 0:
                        x += L
                    vals.append(x)
                
                # Verify distinctness and bounds
                if len(set(vals)) == n and max(vals) <= 1000000:
                    return vals, S
        return None

    # Binary search for minimal W
    # Lower bound: 9 (distinct integers)
    # Upper bound: L (always possible to pick 1 multiple for each, span < L)
    lo = 9
    hi = L
    best_result = None
    
    # We want the minimal W.
    # Check if W=9 is possible, then 10, etc?
    # Binary search is better if the predicate is monotonic.
    # "Exists SDR in window of size W" is monotonic in W.
    # If it exists for W, it exists for W+1.
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        result = exists_sdr_for_W(mid)
        if result is not None:
            best_result = result[0]
            hi = mid
        else:
            lo = mid + 1
            
    if best_result is None:
        # Fallback: try to find any solution
        # Use W = L
        result = exists_sdr_for_W(L)
        if result:
            best_result = result[0]
        else:
            # Absolute fallback
            reps = [int((i + 1) * L) for i in range(n)]
            best_result = reps

    return [int(v) for v in best_result]
