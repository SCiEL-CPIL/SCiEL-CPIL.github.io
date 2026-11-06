import time

def construct():
    P = 1009
    M = 5
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Precompute quadratic residues and their roots
    # residue_bitset: bit i is 1 if i is a non-zero quadratic residue mod P
    # roots: list where roots[r] is a root x such that x^2 % P == r
    residue_bitset = 0
    roots = [0] * P  # 0 means no root found yet (or residue is 0)

    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            residue_bitset |= (1 << r)
            if roots[r] == 0:
                roots[r] = x

    # Mask for M consecutive bits
    mask = (1 << M) - 1

    # Search for the smallest s in [1, P-M]
    # s+i must be <= P-1 to avoid wrap-around issues in the problem statement context
    # The problem states 1 <= s <= 1004 and 1 <= ri <= 1008.
    # If s=1004, s+4=1008. 1008 < 1009. So no wraparound in indices.
    
    best_result = None

    for s in range(1, P - M + 1):
        # Check if s, s+1, ..., s+M-1 are all residues
        # Shift bitset right by s, so bit 0 corresponds to s.
        # Check if lowest M bits are all 1.
        if ((residue_bitset >> s) & mask) == mask:
            # Found minimal s
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}
        
        # Time guard (unlikely to trigger, but required by directive)
        if time.time() > deadline:
             if best_result is None:
                 raise TimeoutError("Search timed out without finding a result")
             return best_result

    raise RuntimeError('No valid run found')
