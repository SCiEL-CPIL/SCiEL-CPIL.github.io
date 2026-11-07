import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Finds an integer m in [0, 1000000] that maximizes f(12, m),
    where f(12, m) is the least positive L such that (m, m+L] contains
    distinct b_1...b_12 with i dividing b_i.
    """
    n = 12
    max_L = 144
    period = 27720  # lcm(1..12)
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    best_m = 0
    best_L = 0
    
    def can_match(m, L):
        """
        Checks if there is a matching for window (m, m+L].
        Returns True if valid distinct b_i exist.
        """
        masks = []
        for i in range(1, n + 1):
            mask = 0
            # v = m + 1 + j, j in 0..L-1
            # (m + 1 + j) % i == 0
            # j % i == (-(m+1)) % i
            target = (-(m + 1)) % i
            # Find all j in [0, L) such that j % i == target
            # j = target, target+i, target+2i, ...
            # Optimization: generate directly
            j = target
            if j >= L:
                # No solution for this i
                return False
            while j < L:
                mask |= (1 << j)
                j += i
            
            masks.append(mask)
        
        # Sort by number of options (most constrained first)
        # We need to keep track of original index or just the masks
        # Since we just need to check existence, we can sort masks by popcount
        masks.sort(key=lambda x: x.bit_count())
        
        # DFS
        # used_mask tracks which positions (0..L-1) are already assigned
        # k is the index in the sorted masks array
        def dfs(k, used_mask):
            if k == n:
                return True
            avail = masks[k] & (~used_mask)
            # Iterate through set bits in avail
            while avail:
                # Get lowest set bit
                lsb = avail & (-avail)
                # Remove it from avail for next iteration
                avail ^= lsb
                # Try assigning this position
                if dfs(k + 1, used_mask | lsb):
                    return True
            return False
        
        return dfs(0, 0)

    # Iterate over one period
    for m in range(period):
        if time.time() - start_time > time_limit:
            break
            
        # Find minimum L for this m
        # We know L is at least 12 (pigeonhole/size)
        # We check L from 1 to max_L
        # Optimization: if best_L is already high, we can start checking from best_L
        # But we need to find the MIN L for this m to compare with best_L
        # Actually, we want to find m that MAXIMIZES the MIN L.
        # So for a candidate m, we find L_min(m).
        # If L_min(m) > best_L, update best_L and best_m.
        
        # We can stop early if we find a match for L <= best_L?
        # No, we need to know the exact L_min(m) to see if it's better.
        # However, if L_min(m) <= best_L, it's not better.
        # So we can check L = 1, 2, ..., best_L.
        # If any of these work, then L_min(m) <= best_L, so this m is not an improvement.
        # We only need to check L > best_L if we want to see if it's better?
        # Wait. If L=1 works, L_min=1.
        # If L=1 fails, L=2 works, L_min=2.
        # We want to find m where L_min is large.
        # So we should check L=1, 2, ... until it succeeds.
        # If it succeeds at L <= best_L, we can stop checking this m.
        # If it fails for all L <= best_L, we continue checking L > best_L.
        
        found_match = False
        for L in range(1, max_L + 1):
            if L <= best_L:
                # If we find a match here, L_min <= best_L, so not better
                if can_match(m, L):
                    found_match = True
                    break
            else:
                # L > best_L
                # If we find a match here, L_min = L > best_L, so this is a new best
                if can_match(m, L):
                    best_L = L
                    best_m = m
                    found_match = True
                    break
                # If L reaches max_L and no match, L_min > max_L (impossible by problem def, or cap)
        
        # If no match found up to max_L, L_min is effectively > max_L (or max_L if we cap)
        # Problem says raw value is at most 144.
        
    return int(best_m)

# EVOLVE-BLOCK-END
