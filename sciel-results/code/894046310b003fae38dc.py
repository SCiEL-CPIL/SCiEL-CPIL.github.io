"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    # Precompute options for each i
    # v is in (m, m+L], so v in {m+1, ..., m+L}
    # v % i == 0
    # Let v = m + 1 + j for j in 0..L-1
    # (m + 1 + j) % i == 0  => j % i == (-(m+1)) % i
    
    options = []
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        # j ranges from 0 to L-1
        # We can iterate j and check j % i == target
        # Or find the first j and step by i
        first_j = target
        if first_j >= L:
            # No solution for this i
            return False
        # Generate mask
        # j = first_j, first_j + i, ... < L
        j = first_j
        while j < L:
            mask |= (1 << j)
            j += i
        
        if mask == 0:
            return False
        # Store (count, mask, i) to sort by constraint
        options.append((mask.bit_count(), mask, i))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    # DFS
    # used_mask tracks which positions j (0..L-1) are taken
    # We need to assign a unique j to each i
    # Since L <= 144, we can use a Python int as a bitmask for used positions if L is small enough?
    # 144 bits is fine for Python int.
    
    def dfs(k, used_mask):
        if k == n:
            return True
        count, mask, i = options[k]
        # Available positions are those in mask but not in used_mask
        avail = mask & ~used_mask
        # Try each available position
        while avail:
            # Get lowest set bit
            lsb = avail & -avail
            avail ^= lsb
            # Try assigning this position to i
            if dfs(k + 1, used_mask | lsb):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] allows matching.
    """
    lo, hi = 1, 144
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    """
    Find m in 0..1000000 maximizing f(12, m).
    Due to periodicity LCM(1..12)=27720, we only search 0..27719.
    """
    best_m = 0
    best_f = 0
    period = 27720
    
    # Deadline guard: 90s limit. This loop should take < 1s.
    start_time = time.time()
    
    for m in range(period):
        if time.time() - start_time > 60:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    return best_m
# EVOLVE-BLOCK-END
