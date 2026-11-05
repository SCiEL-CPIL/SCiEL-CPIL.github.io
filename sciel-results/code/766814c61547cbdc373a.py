import time

P = 1009
M = 5
DEADLINE_MARGIN = 2.0  # seconds to stop before the 90s wall-clock limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    # Build a boolean array is_qr where is_qr[v] is True if v is a nonzero quadratic residue mod P.
    # Also store one root for each residue for the output.
    is_qr = [False] * P
    roots = [0] * P  # roots[v] = x such that x^2 % P == v, 0 if not a QR
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            # We only need one root. The first one found is fine.
            # Storing it allows us to return it without searching again.
            if roots[r] == 0:
                roots[r] = x

    # Create a bitmask for efficient checking.
    # Bit i is 1 if i is a QR.
    mask = 0
    for i in range(1, P):
        if is_qr[i]:
            mask |= (1 << i)

    # The target pattern for 5 consecutive QRs starting at s.
    # We want bits s, s+1, s+2, s+3, s+4 to all be 1.
    # Pattern: 0b11111 = 31.
    # We check (mask >> s) & 31 == 31.
    pattern = 31

    # Search for the minimum s such that s, s+1, ..., s+4 are all nonzero QRs.
    # s must be in [1, P - M] to ensure s+4 <= P-1 (no wraparound, no zero).
    # Range of s: 1 to 1009 - 5 = 1004.
    
    for s in range(1, P - M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all QRs using bitwise operation
        if (mask >> s) & pattern == pattern:
            # Found the minimal s
            result_roots = [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            
            # Validation
            for i in range(M):
                r = result_roots[i]
                if (r * r) % P != s + i:
                    # Fallback: find a valid root if the stored one is somehow invalid (shouldn't happen)
                    # Search for a root
                    found = False
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            result_roots[i] = x
                            found = True
                            break
                    if not found:
                        raise RuntimeError(f"No root found for {s+i}")
            
            # Return immediately since we found the minimal s
            return {'start': s, 'roots': result_roots}
        
        # Check deadline
        if time.time() > deadline:
            break

    # If we didn't find a run in the first pass (or timed out), 
    # try a descending search as a fallback (though this should not happen for P=1009)
    for s in range(P - M, 0, -1):
        if (mask >> s) & pattern == pattern:
            result_roots = [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            for i in range(M):
                r = result_roots[i]
                if (r * r) % P != s + i:
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            result_roots[i] = x
                            break
            return {'start': s, 'roots': result_roots}

    raise RuntimeError("No run found within deadline or search space")
