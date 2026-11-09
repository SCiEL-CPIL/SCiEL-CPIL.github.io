import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5
    
    # Precompute a root for each non-zero quadratic residue modulo p.
    # We store the smallest root found for each residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the root. If multiple roots exist, we keep the first one found (smallest x).
            if r not in roots:
                roots[r] = x
        # Check deadline occasionally, though this loop is very fast
        if time.time() > deadline:
            # Fallback: return a known valid late run or raise
            # Since we haven't found the min yet, we can't return a min.
            # But precomputation is < 1ms, so this is unlikely.
            raise TimeoutError("Computation exceeded deadline during precomputation")

    # Search for the smallest s such that s, s+1, ..., s+m-1 are all quadratic residues.
    # s must be in 1..1004 to ensure s+m-1 <= 1008 (no wraparound, no zero).
    for s in range(1, p - m + 1):
        # Check if all s+i are in roots
        valid = True
        for i in range(m):
            if (s + i) not in roots:
                valid = False
                break
        
        if valid:
            # Construct the result
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
        
        if time.time() > deadline:
             # If we run out of time, we might not have found the minimum.
             # However, the search is O(p), so it's extremely fast.
             # If this happens, it's an unexpected system issue.
             raise TimeoutError("Computation exceeded deadline during search")

    # If no run is found (should not happen for p=1009, m=5), raise an error.
    raise RuntimeError("No valid run found")
# EVOLVE-BLOCK-END
