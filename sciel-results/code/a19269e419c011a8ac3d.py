"""Rewritten program: minimal run-start search with deadline guard."""
# EVOLVE-BLOCK-START
import time

_P = 1009
_M = 5

def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before the 90s kill

    # Precompute one valid root (1..1008) for every nonzero quadratic residue.
    roots = {}
    for x in range(1, _P):
        v = (x * x) % _P
        if v == 0:
            continue
        if v not in roots:
            roots[v] = x

    # Ascending search for the smallest feasible start s.
    # Feasible s: s in [1,1004] and s+i in roots for i=0..4 (no wraparound, no zero).
    for s in range(1, _P - _M + 1):
        if s + (_M - 1) > _P - 1:
            break
        ok = True
        for i in range(_M):
            if (s + i) not in roots:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
        if time.time() > deadline:
            break

    # Fallback: descending search to guarantee a valid result if ascending failed early.
    for s in range(_P - _M, 0, -1):
        ok = True
        for i in range(_M):
            if (s + i) not in roots:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [roots[s + i] for i in range(_M)]}
        if time.time() > deadline:
            break

    raise RuntimeError('No run found within budget')
# EVOLVE-BLOCK-END
