"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5
_TIMEOUT = 60.0


def construct():
    """
    Return {'start': s, 'roots': [r0, ..., r4]} where:
      - 1 <= s <= 1004
      - 1 <= ri <= 1008
      - ri**2 mod 1009 == s + i for i in 0..4
      - s is minimized
      - zero residues and wraparound are excluded
    """
    start_time = time.time()
    
    # Build a bitset where bit i is set if i is a non-zero quadratic residue mod P
    # and a mapping from each non-zero quadratic residue to one witness root in [1, P-1]
    roots = [0] * _P
    bitset = 0
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            bitset |= (1 << r)
            if roots[r] == 0:
                roots[r] = x
    
    # Ascending search to find the minimum s using bitwise check
    # Since s >= 1 and s <= P - M = 1004, we have s+i <= 1008 < 1009
    # and s+i >= 1, so no wraparound and no zero residue is possible
    mask = 0b11111  # 5 consecutive bits
    
    for s in range(1, _P - _M + 1):
        if time.time() - start_time > _TIMEOUT - 1.0:
            # Safety guard; this loop is very fast so we should not hit this
            break
        
        # Check that s, s+1, ..., s+4 are all quadratic residues
        if (bitset >> s) & mask == mask:
            return {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
    
    raise RuntimeError('No run found')
