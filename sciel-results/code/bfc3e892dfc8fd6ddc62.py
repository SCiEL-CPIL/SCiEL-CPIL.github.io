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
    
    # Build a mapping from each non-zero quadratic residue to one witness root in [1, P-1]
    roots = {}
    for x in range(1, _P):
        if time.time() - start_time > _TIMEOUT - 1.0:
            # Safety guard, though this loop is very fast
            break
        r = (x * x) % _P
        if r != 0 and r not in roots:
            roots[r] = x

    # Ascending search to find the minimum s
    best_result = None
    for s in range(1, _P - _M + 1):
        if time.time() - start_time > _TIMEOUT - 1.0:
            break
            
        # Check that s, s+1, ..., s+4 are all in [1, P-1] (no wraparound, no zero)
        # Since s >= 1 and s <= P - M = 1004, we have s+i <= 1004+4 = 1008 < 1009
        # and s+i >= 1, so no wraparound and no zero residue is possible
        if all((s + i) in roots for i in range(_M)):
            best_result = {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
            # Since we are scanning in ascending order, the first one found is the minimum
            return best_result

    if best_result is not None:
        return best_result

    raise RuntimeError('No run found')
