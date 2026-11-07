<<<<<<< SEARCH
import time
import multiprocessing as mp

def _can_match_fast(m, L, n=12):
    # Compute bitmasks for each index i (1..n)
    # Bit k (0-indexed) represents value m + k + 1
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        # k goes from 1 to L
        # Value is m + k
        # Condition: (m + k) % i == 0
        # k % i == (-m) % i
        rem = (-m) % i
        for k in range(1, L + 1):
            if k % i == rem:
                mask |= (1 << (k - 1))
        masks[i] = mask
        
    # Check if any mask is 0
    for i in range(1, n + 1):
        if masks[i] == 0:
            return False
            
    # Sort indices by number of choices (popcount)
    indices = list(range(1, n + 1))
    indices.sort(key=lambda x: bin(masks[x]).count('1'))
    
    # Recursive backtracking
    # used_mask: bitmask of used offsets (0..L-1)
    def dfs(idx, used_mask):
        if idx == n:
            return True
        i = indices[idx]
        candidates = masks[i] & ~used_mask
        while candidates:
            # Get lowest set bit
            bit = candidates & -candidates
            candidates -= bit
            if dfs(idx + 1, used_mask | bit):
                return True
        return False
        
    return dfs(0, 0)
=======
import time
import multiprocessing as mp

def _can_match_fast(m, L, n=12):
    # Compute bitmasks for each index i (1..n)
    # Bit k (0-indexed) represents value m + k
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        rem = (-m) % i
        mask = 0
        # k goes from 1 to L, value is m+k, need (m+k)%i==0 => k%i == rem
        k = rem if rem != 0 else i
        while k <= L:
            mask |= (1 << (k - 1))
            k += i
        masks[i] = mask
        
    # Check if any mask is 0
    for i in range(1, n + 1):
        if masks[i] == 0:
            return False
            
    # Sort indices by number of choices (popcount)
    indices = list(range(1, n + 1))
    indices.sort(key=lambda x: masks[x].bit_count())
    
    # Recursive backtracking with forward checking
    def dfs(idx, used_mask):
        if idx == n:
            return True
        i = indices[idx]
        candidates = masks[i] & ~used_mask
        if candidates == 0:
            return False
        while candidates:
            # Get lowest set bit
            bit = candidates & -candidates
            candidates -= bit
            # Forward checking: ensure next variable still has candidates
            next_i = indices[idx + 1]
            if (masks[next_i] & ~(used_mask | bit)) == 0:
                continue
            if dfs(idx + 1, used_mask | bit):
                return True
        return False
        
    return dfs(0, 0)
>>>>>>> REPLACE

Now let me also tighten the deadline and add a safety check:

