"""Find the least run-start s for 5 consecutive non-zero quadratic residues mod 1009."""

_P = 1009
_M = 5


def _build_table():
    """Map each non-zero quadratic residue to one witness root in [1, P-1]."""
    root = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root:
            root[r] = x
    return root


def _verify(result):
    s = result['start']
    roots = result['roots']
    if not (1 <= s <= _P - _M):
        return False
    if len(roots) != _M:
        return False
    for i in range(_M):
        v = s + i
        if v == 0 or v >= _P:
            return False
        ri = roots[i]
        if not (1 <= ri <= _P - 1):
            return False
        if (ri * ri) % _P != v:
            return False
    return True


def construct():
    # Build the quadratic residue table: residue -> smallest root
    qr_roots = _build_table()

    # We need s, s+1, s+2, s+3, s+4 all to be non-zero quadratic residues mod 1009.
    # Since 1009 is prime, non-zero QRs are exactly the values x^2 mod 1009 for x in 1..1008.
    # There are (1009-1)/2 = 504 non-zero QRs.
    # We search for the smallest s in [1, 1004] such that s..s+4 are all in qr_roots.

    # Direct ascending search
    for s in range(1, 1005):
        ok = True
        for i in range(_M):
            v = s + i
            if v == 0 or v >= _P or v not in qr_roots:
                ok = False
                break
        if ok:
            result = {'start': s, 'roots': [qr_roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    # Fallback: backward search (shouldn't happen for this fixed instance)
    for s in range(1004, 0, -1):
        ok = True
        for i in range(_M):
            v = s + i
            if v == 0 or v >= _P or v not in qr_roots:
                ok = False
                break
        if ok:
            result = {'start': s, 'roots': [qr_roots[s + i] for i in range(_M)]}
            if _verify(result):
                return result

    raise RuntimeError('No run found')
