"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_P = 1009
_M = 5
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0  # safety margin seconds


def construct():
    """Find minimal start s for 5 consecutive non-zero quadratic residues mod 1009."""
    # Build root map: residue -> smallest root r in [1, P-1] with r^2 % P == residue.
    # Zero residue is excluded, so we skip residue 0.
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Build a bitset where bit i is 1 iff i (1..1008) is a non-zero quadratic residue.
    # We use bit positions 1..1008 (bit 0 unused).
    mask = 0
    for v in range(1, _P):
        if root[v] != 0:
            mask |= (1 << v)

    # To check if s, s+1, ..., s+4 are all residues, we need bits s through s+4 all set.
    # Create a pattern with 5 consecutive bits set: 0b11111 = 31.
    # Shift this pattern to position s: pattern << s.
    # Check: (mask & (pattern << s)) == (pattern << s)
    # This is equivalent to: (mask >> s) & 31 == 31

    # Search for minimum s in 1..1004 such that the condition holds.
    # Use the bitset check which is O(1) per candidate s.
    max_s = _P - _M  # 1004
    for s in range(1, max_s + 1):
        # Check if 5 consecutive bits starting at position s are all set
        if (mask >> s) & 31 == 31:
            roots = [root[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}

    # Fallback (should not happen for p=1009), but guard with deadline.
    if time.time() > _DEADLINE - _MARGIN:
        # Should not reach here; return best effort is impossible, but avoid hang.
        pass
    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
# EVOLVE-BLOCK-END
