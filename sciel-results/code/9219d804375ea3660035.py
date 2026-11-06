"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""
import time

_P = 1009
_M = 5


def _build_table():
    """Build a mapping from each non-zero quadratic residue to one witness root in [1, P-1]."""
    roots = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in roots:
            roots[r] = x
    return roots


def _verify(start, roots):
    """Verify that the given start and roots satisfy the constraints."""
    if not (1 <= start <= _P - _M):
        return False
    if len(roots) != _M:
        return False
    for i in range(_M):
        val = start + i
        if not (1 <= val <= _P - 1):
            return False
        if (roots[i] ** 2) % _P != val:
            return False
    return True


def construct():
    """
    Return {'start': s, 'roots': [r0, ..., r4]} where:
      - 1 <= s <= 1004
      - 1 <= ri <= 1008
      - ri**2 mod 1009 == s + i for i in 0..4
      - s is minimized
      - zero residues and wraparound are excluded
    """
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin before 90s kill

    roots = _build_table()

    # Ascending search to find the minimum s
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            break
        if all((s + i) in roots for i in range(_M)):
            result_roots = [roots[s + i] for i in range(_M)]
            if _verify(s, result_roots):
                return {'start': s, 'roots': result_roots}

    raise RuntimeError('No run found')
