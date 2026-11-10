import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of the divisors
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    LCM = 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Fallback: the divisors themselves form a valid set.
    # Max = 39, Min = 6, Span = 33.
    best_result = divisors[:]
    best_span = max(divisors) - min(divisors)
    
    # We want to minimize the span W = max - min.
    # The minimum possible span for 10 distinct integers is 9.
    # We iterate W from 9 up to best_span - 1.
    
    # Precompute divisors sorted by their "constraint tightness" for backtracking?
    # Actually, we should sort indices by the number of candidates in the current window
    # at each step of backtracking, but that's dynamic.
    # Static sort: Divisors with larger values have fewer multiples in a small window.
    # Let's sort divisors descending for initial ordering in backtracking to fail fast?
    # No, the number of candidates depends on L.
    
    def check_window(W, L):
        """
        Checks if there is a valid assignment of distinct multiples for the divisors
        within the window [L, L+W].
        Returns a list of chosen values or None.
        """
        # 1. Generate candidates for each divisor
        # candidates[i] is a list of offsets (0 to W) such that L + offset is divisible by divisors[i]
        candidates = []
        for d in divisors:
            # First multiple >= L
            k = (L + d - 1) // d
            first = k * d
            if first > L + W:
                return None
            cand = []
            x = first
            while x <= L + W:
                cand.append(x - L)
                x += d
            candidates.append(cand)
            
        # 2. Sort indices by number of candidates (ascending) to find conflicts early
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        # 3. Backtracking with bitmask
        # If W is small enough (W+1 <= 64), we can use an integer bitmask for used positions.
        # Since we are searching for small W, this is likely always true.
        if W + 1 > 64:
            # Fallback to set if W is large (unlikely in this search)
            used = set()
            choices = [0] * n
            
            def bt(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for off in candidates[i]:
                    val = L + off
                    if val not in used:
                        used.add(val)
                        choices[i] = off
                        if bt(idx + 1):
                            return True
                        used.remove(val)
                return False
                
            if bt(0):
                return [L + choices[i] for i in range(n)]
            return None
        else:
            used = 0
            choices = [0] * n
            
            def bt(idx):
                nonlocal used
                if idx == n:
                    return True
                i = indices[idx]
                for off in candidates[i]:
                    bit = 1 << off
                    if not (used & bit):
                        used |= bit
                        choices[i] = off
                        if bt(idx + 1):
                            return True
                        used &= ~bit
                return False
                
            if bt(0):
                return [L + choices[i] for i in range(n)]
            return None

    # Search for minimal W
    for W in range(9, best_span):
        if time.time() - start_time > timeout:
            break
            
        # For this W, iterate L from 1 to LCM
        # Optimization: We can skip L values that are equivalent in terms of "phase"
        # but since LCM is the period, we must check all.
        # However, we can break as soon as we find a valid L.
        
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
                
            # Quick feasibility check:
            # For each divisor d, is there a multiple in [L, L+W]?
            # If not, skip.
            feasible = True
            for d in divisors:
                # Check if interval [L, L+W] contains a multiple of d
                # ceil(L/d) * d <= L + W
                if (L + d - 1) // d * d > L + W:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            result = check_window(W, L)
            if result is not None:
                # Verify
                if len(set(result)) == n:
                    if all(result[i] % divisors[i] == 0 for i in range(n)):
                        if all(1 <= x <= 1000000 for x in result):
                            return result
                            
    # If no improvement found, return the initial best
    return best_result
