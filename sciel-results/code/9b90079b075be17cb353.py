import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030  # LCM of divisors
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute nothing, generate candidates on the fly for speed
    
    def has_sdr(sets):
        """
        sets: list of lists of integers
        Returns True if there is a system of distinct representatives.
        Uses backtracking with MRV heuristic.
        """
        # Sort indices by set size
        indices = list(range(len(sets)))
        indices.sort(key=lambda i: len(sets[i]))
        
        # If any set is empty, no SDR
        if any(len(sets[i]) == 0 for i in indices):
            return False, None
            
        used = set()
        assignment = [None] * len(sets)
        
        def dfs(idx):
            if idx == len(sets):
                return True
            i = indices[idx]
            for val in sets[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if dfs(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if dfs(0):
            return True, assignment
        return False, None

    # Search for minimal W
    for W in range(10, 500):
        if time.time() > deadline:
            break
            
        for S in range(L):
            if time.time() > deadline:
                break
                
            current_sets = []
            valid = True
            for i in range(n):
                d = divisors[i]
                # Find first multiple of d >= S
                # p = ceil(S/d) * d
                p = ((S + d - 1) // d) * d
                
                candidates = []
                # We need p such that (p - S) % L <= W
                # Since p >= S, (p - S) is just p - S.
                # But p might wrap around?
                # If we generate p, p+d, ... we might exceed S+W.
                # Also, if S+W >= L, we might have candidates from the "other side" (0 to S+W-L).
                
                # Case 1: p in [S, min(S+W, L-1)]
                limit1 = min(S + W, L - 1)
                while p <= limit1:
                    candidates.append(p)
                    p += d
                    
                # Case 2: wrap around. If S + W >= L, then [0, S+W-L] are also valid.
                if S + W >= L:
                    limit2 = S + W - L
                    # Smallest multiple of d >= 0 is 0
                    p2 = 0
                    while p2 <= limit2:
                        # Avoid duplicates if L is multiple of d (which it is)
                        # If p2 == some p from case 1?
                        # p in case 1 is >= S. p2 < S (since limit2 < S). So no overlap.
                        candidates.append(p2)
                        p2 += d
                        
            if not candidates:
                 valid = False
                 continue
                 
            current_sets.append(candidates)
            
            if not valid:
                continue
                
            # Check SDR
            found, assignment = has_sdr(current_sets)
            
            if found:
                # Construct values
                vals = []
                for i in range(n):
                    p = assignment[i]
                    # p is in [0, L)
                    # We need positive integer.
                    # If p == 0, use L. Else use p.
                    # Wait, the window was [S, S+W].
                    # If p >= S, value is p.
                    # If p < S, value is p + L.
                    if p >= S:
                        x = p
                    else:
                        x = p + L
                        
                    # Ensure x is positive
                    if x <= 0:
                        x += L
                        
                    vals.append(x)
                    
                # Check distinctness and bounds
                # They are distinct by SDR construction on residues modulo L.
                # Map to [1, L] or [1, 2L]?
                # My logic: if p >= S, x=p. if p < S, x=p+L.
                # Since p is in [0, L), x is in [1, 2L).
                # Are they distinct?
                # If p1 != p2 mod L, then x1 != x2 mod L?
                # x1 = p1 + k1*L, x2 = p2 + k2*L.
                # If x1 == x2, then p1 - p2 = (k2-k1)L.
                # Since p1, p2 in [0, L), p1-p2 in (-L, L).
                # So p1-p2 must be 0. So p1=p2.
                # So yes, distinct.
                
                # Check max <= 1000000
                if max(vals) <= 1000000:
                    return [int(v) for v in vals]
                else:
                    # Should not happen for small W
                    # But if it does, we can shift down?
                    # No, we want minimal W.
                    # If max > 10^6, we can subtract L?
                    # But we need positive.
                    # Since L=30030, and W is small, max is ~30000.
                    # So this check is just a safety net.
                    pass

    # Fallback
    reps = [int((i + 1) * L) for i in range(n)]
    return reps

