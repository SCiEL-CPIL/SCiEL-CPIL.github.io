"""Find the smallest start s such that s, s+1, s+2, s+3, s+4 are all quadratic residues mod 1009."""
import time

def construct():
    start_time = time.time()
    p = 1009
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute: for each quadratic residue r in 1..p-1, store one root x
    # such that x*x % p == r. Also build a bitmask of which values are residues.
    roots = [0] * p  # roots[r] = x if r is a QR, else 0
    is_residue = [False] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and not is_residue[r]:
            is_residue[r] = True
            roots[r] = x

    # Build a Python int as a bitset: bit i set means i is a quadratic residue (1..1008)
    bitset = 0
    for i in range(1, p):
        if is_residue[i]:
            bitset |= (1 << i)

    # Compute the AND of bitset shifted by 0,1,2,3,4 positions.
    # Bit s in combined is set iff bits s, s+1, s+2, s+3, s+4 are all set in bitset.
    combined = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)

    # Mask out bit 0 (s=0 is not valid, we need s>=1)
    combined &= ~1

    # Find the lowest set bit in combined (that's >= bit 1)
    if combined != 0:
        lowest = combined & (-combined)
        s = lowest.bit_length() - 1
        if 1 <= s <= 1004:
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}

    # Fallback: linear scan (should never be needed for p=1009)
    if time.time() > deadline:
        raise RuntimeError("Deadline exceeded")

    for s in range(1, 1005):
        if all(is_residue[s + i] for i in range(5)):
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}

    raise RuntimeError('No valid run found')
