import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # Build root map: residue -> smallest root r in [1, P-1] with r^2 % P == residue
    # Zero residue is excluded
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Build bitset: bit i set iff i is a non-zero quadratic residue
    bitset = 0
    for i in range(1, _P):
        if root[i] != 0:
            bitset |= (1 << i)

    # Scan s from 1 upward to find the MINIMUM start where 5 consecutive residues exist
    # s must satisfy 1 <= s <= 1004 (so s+4 <= 1008, no wraparound)
    max_s = _P - 5  # 1004
    for s in range(1, max_s + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in the bitset
        # (bitset >> s) shifts so bit s becomes bit 0, then mask 0b11111 checks 5 bits
        if ((bitset >> s) & 0b11111) == 0b11111:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    # Fallback: should not happen for p=1009, but guard against timeout
    if time.time() > _DEADLINE - _MARGIN:
        raise RuntimeError('Deadline exceeded without finding a 5-run')
    
    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
