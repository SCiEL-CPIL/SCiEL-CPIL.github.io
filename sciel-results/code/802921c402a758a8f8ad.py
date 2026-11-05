import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [s, s+l-1] subset of [1, 1008]
    such that for ALL x, y in the interval (including x=y), x*y % 1009 != 1.
    Maximize l.
    
    Strategy: Two-pointer sliding window with precomputed inverse map.
    An interval is valid iff no element x in it has its modular inverse inv[x]
    also in it (including the self-inverse cases x=1, x=1008).
    """
    p = 1009
    n = 1008
    deadline = time.time() + 50  # safety margin
    
    # Precompute modular inverses: inv[x] = x^{-1} mod p
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Two-pointer sliding window
    # L = left boundary of current window
    # For each R, we ensure [L, R] is valid by pushing L past any conflict
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverse residues (1 and 1008) cannot appear in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict.
        # Push L past invR to resolve.
        if L <= invR < R:
            L = invR + 1
        
        # Window [L, R] is now valid by construction:
        # - R's inverse is outside [L, R]
        # - All previous elements in [L, R-1] were validated when added
        # - Moving L forward only removes elements, never creates new conflicts
        if L <= R:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L
    
    # Final validation pass: verify the found interval is truly valid
    # This guards against any subtle bug in the sliding window logic
    if best_len > 0:
        end = best_start + best_len - 1
        valid = True
        for x in range(best_start, end + 1):
            ix = inv[x]
            if ix >= best_start and ix <= end:
                valid = False
                break
        
        if not valid:
            # Fallback: brute-force search for longest valid interval
            # O(n^2) but n=1008 so this is fast
            best_len = 0
            best_start = 1
            for s in range(1, n + 1):
                if time.time() > deadline:
                    break
                # Find the maximum valid end for start s
                e = s
                while e <= n:
                    if inv[e] >= s and inv[e] <= e:
                        break
                    e += 1
                # e is the first invalid position, so valid interval is [s, e-1]
                if e > s:
                    curr_len = e - s
                    if curr_len > best_len:
                        best_len = curr_len
                        best_start = s
    
    # Ensure we return a nonempty interval
    if best_len < 1:
        # Find any single valid element
        for x in range(2, n):
            if x != 1 and x != 1008:
                if not (inv[x] >= x and inv[x] <= x):  # x is not self-inverse
                    best_start = x
                    best_len = 1
                    break
        if best_len < 1:
            best_start = 2
            best_len = 1
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
