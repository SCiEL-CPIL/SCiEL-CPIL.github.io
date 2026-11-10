import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to determine periodicity
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # Primes: 2, 3, 5, 7, 11, 13
    lcm_val = 30030
    
    def get_candidates(S, L, d):
        """Get multiples of d in [S, S+L] that are <= max_val"""
        k = (S + d - 1) // d
        start_mult = k * d
        candidates = []
        val = start_mult
        limit = min(S + L, max_val)
        while val <= limit:
            if val > 0:
                candidates.append(val)
            val += d
        return candidates
    
    def check_assignment(candidates):
        """
        Check if we can assign distinct values from candidates[i] to position i.
        Returns a list of assigned values or None.
        Uses MRV heuristic for faster pruning.
        """
        assigned = [0] * n
        used = set()
        
        # Sort indices by number of candidates (MRV heuristic)
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    assigned[i] = val
                    used.add(val)
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assigned[i] = 0
            return False
        
        if backtrack(0):
            return assigned
        return None
    
    def exists_window(L):
        """
        Check if there exists a window [S, S+L] with a valid distinct assignment.
        Returns the assignment if found, else None.
        Uses periodicity: only need to check S in [1, LCM].
        """
        for S in range(1, lcm_val + 1):
            if time.time() > deadline:
                return None
            
            candidates = []
            valid = True
            for d in divisors:
                c = get_candidates(S, L, d)
                if not c:
                    valid = False
                    break
                candidates.append(c)
            
            if not valid:
                continue
                
            res = check_assignment(candidates)
            if res is not None:
                # Validate: all distinct, positive, <= max_val, divisible by corresponding divisor
                if len(set(res)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(res)):
                    return res
        
        return None
    
    # Strategy: Linear search from smallest possible W (n-1 = 9) upwards
    # This guarantees finding the minimum W if we can afford the time.
    # If linear search is too slow, we fall back to binary search or return best found.
    
    best_solution = None
    best_W = float('inf')
    
    # Try small W values first. 
    # For each W, we scan S in [1, LCM].
    # If a solution is found for W, it's the minimum since we go in increasing order.
    
    # Estimate cost: 
    # For each W, we iterate S from 1 to 30030.
    # For each S, we generate candidates (cheap) and run backtracking.
    # Backtracking is fast for small W (few candidates).
    
    # Let's try W from 9 to 100 first.
    # If not found, try binary search between 100 and some upper bound.
    
    for W in range(9, 101):
        if time.time() > deadline:
            break
            
        res = exists_window(W)
        if res is not None:
            best_solution = res
            best_W = W
            break  # Found minimum W, stop
    
    # If not found in small range, try binary search for a larger W
    if best_solution is None:
        low = 101
        high = 10000
        
        # Binary search for the smallest W that has a solution
        # But we want the *minimum* W. Binary search finds *a* W.
        # Since "exists" is monotonic (if exists for W, exists for W+1? 
        # Actually, if there's a solution in [S, S+W], the same values are in [S, S+W+1].
        # So yes, if exists(W) is True, exists(W+1) is True.
        # Thus, we can binary search for the smallest W where exists(W) is True.
        
        while low <= high:
            if time.time() > deadline:
                break
                
            mid = (low + high) // 2
            res = exists_window(mid)
            if res is not None:
                best_solution = res
                best_W = mid
                high = mid - 1
            else:
                low = mid + 1
    
    # If still no solution (very unlikely), use a greedy fallback
    if best_solution is None:
        # Greedy: pick closest multiples to a center point
        # Center around LCM/2 or just 1
        # Let's try to construct a valid solution greedily
        # Sort divisors by size? Or just pick the first available multiple in a range?
        # Simple fallback: return the divisors themselves if they are distinct (they are)
        # But they might not be optimal.
        # Let's try a quick greedy around LCM/2
        center = lcm_val // 2
        candidates = []
        used = set()
        for d in divisors:
            k = center // d
            # Try a few offsets
            found = False
            for offset in range(-10, 11):
                val = (k + offset) * d
                if val > 0 and val <= max_val and val not in used:
                    candidates.append(val)
                    used.add(val)
                    found = True
                    break
            if not found:
                # Fallback to just d
                candidates.append(d)
        
        # Validate
        if len(set(candidates)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(candidates)):
            best_solution = candidates
        else:
            # Emergency: return divisors
            best_solution = divisors[:]
    
    # Ensure types are Python int
    return [int(x) for x in best_solution]
