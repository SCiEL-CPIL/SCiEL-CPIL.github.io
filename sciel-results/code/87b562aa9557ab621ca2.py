"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Returns the start index and the corresponding smallest positive square roots.

Uses a bitset-based approach for fast pattern matching.
"""
import time

def construct():
    p = 1009
    m = 5
    deadline = time.time() + 80.0  # 80s safety margin

    # Boolean array to check if a number is a quadratic residue
    is_res = [False] * p
    # Array to store the smallest positive root for each residue
    root_map = [0] * p

    # Precompute residues and roots
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and not is_res[r]:
            is_res[r] = True
            root_map[r] = x

    # Build a bitset where bit i is 1 iff i is a quadratic residue (1 <= i <= p-1)
    # We use bit position i for value i (1-indexed), so bit 0 is unused
    mask = 0
    for i in range(1, p):
        if is_res[i]:
            mask |= (1 << i)

    # Find all positions s where s, s+1, s+2, s+3, s+4 are all residues
    # Using bitwise AND of shifted masks
    # A valid start s means bits s, s+1, s+2, s+3, s+4 are all set
    # (mask >> k) has bit i set iff mask has bit i+k set
    # So (mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)) 
    # has bit s set iff bits s, s+1, s+2, s+3, s+4 are all set in mask
    valid_starts = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)

    # Find the smallest s (1 <= s <= p - m) such that bit s is set in valid_starts
    # We need s >= 1 and s <= p - m = 1004
    # Check bits from position 1 upward
    for s in range(1, p - m + 1):
        if valid_starts & (1 << s):
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No consecutive run of quadratic residues found")
