"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses in [1, 1008]
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x where x*x % p == 1
    # These are x=1 and x=p-1=1008
    # For any interval containing 1 or 1008, the pair (1,1) or (1008,1008) gives 1*1%1009=1 or 1008*1008%1009=1
    # So intervals cannot contain 1 or 1008
    
    best_len = 0
    best_start = 1
    
    # Try all possible intervals
    for start in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If start is a self-inverse, skip (can't include it)
        if start == 1 or start == 1008:
            continue
            
        # For each start, find the maximum end such that no x,y in [start, end] have x*y % p == 1
        # We need to check all pairs. But we can be smarter.
        # For each x in [start, ...], if inv[x] is also in the interval, it's invalid.
        # So the interval must not contain both x and inv[x] for any x.
        # Also x != inv[x] except for self-inverses (1 and p-1=1008), which are forbidden.
        
        # Let's build the interval greedily or just check all ends.
        # Since n is small (1008), we can just check all ends.
        
        # Track which values are "blocked" - if we add x, we can't have inv[x]
        # Actually, let's just iterate end from start to n and check validity incrementally
        
        valid = True
        # We'll track a set of values in the current interval
        # For efficiency, let's precompute for each start the maximum valid end
        
        # Alternative: for each start, extend end as far as possible
        # When we add a new element y, check if inv[y] is already in [start, y-1]
        # If yes, we can't extend further (well, we could skip y, but we want contiguous interval)
        
        # Wait, the interval must be contiguous [start, end]. So we can't skip elements.
        # So for a fixed start, we extend end until we find a conflict.
        
        # For each end from start to n:
        #   Check if adding 'end' creates a conflict with any existing element in [start, end-1]
        #   A conflict exists if there exists x in [start, end-1] such that x * end % p == 1
        #   i.e., x == inv[end]
        #   So conflict if inv[end] >= start and inv[end] < end (i.e., inv[end] is in [start, end-1])
        #   Also, we need to check if end itself is self-inverse: end*end % p == 1
        #   Self-inverses are 1 and 1008, already handled.
        
        # Actually, we also need to check pairs among elements already in [start, end-1],
        # but if [start, end-1] was valid, then all pairs within it are fine.
        # We only need to check the new element 'end' against all elements in [start, end-1].
        # The only element that could conflict with 'end' is inv[end].
        # So the check is: is inv[end] in [start, end-1]?
        
        # Let's verify: if x*y % p == 1, then y = inv[x]. So for a fixed y=end,
        # the only x that conflicts is x = inv[end].
        # So we just need to check if inv[end] is in [start, end-1].
        
        # But wait, we also need to make sure that when we add 'end', it doesn't conflict
        # with itself: end * end % p != 1. This is true unless end is 1 or 1008.
        
        max_end = start - 1
        for end in range(start, n + 1):
            if time.time() > deadline:
                break
                
            # Check if end is self-inverse
            if end == 1 or end == 1008:
                break  # Can't include this, and since interval is contiguous, we stop
                
            # Check if inv[end] is in [start, end-1]
            ie = inv[end]
            if start <= ie < end:
                break  # Conflict found, can't extend further
            
            # No conflict, so [start, end] is valid
            max_end = end
        
        length = max_end - start + 1
        if length > best_len:
            best_len = length
            best_start = start
    
    if best_len == 0:
        # Fallback: return a small valid interval
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
