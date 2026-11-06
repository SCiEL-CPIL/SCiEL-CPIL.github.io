import time

_P = 1009
_M = 5

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Precompute quadratic residues and one root for each
    # is_residue[i] is True if i is a non-zero quadratic residue mod _P
    # root_of[i] stores a witness x such that x*x % _P == i
    is_residue = [False] * (_P + 1)
    root_of = [0] * (_P + 1)
    
    # We only need to iterate x up to P//2 to cover all residues,
    # but iterating all is fine and simpler.
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and not is_residue[r]:
            is_residue[r] = True
            root_of[r] = x
            
    # Ascending search to find the minimum s
    # s ranges from 1 to 1004 (so s+4 <= 1008, avoiding wraparound)
    # We want the smallest s, so we stop at the first match.
    for s in range(1, _P - _M + 1):
        if all(is_residue[s + i] for i in range(_M)):
            roots = [root_of[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}
            
    # Fallback if no run found (should not happen for p=1009, m=5)
    raise RuntimeError('No baseline run found')
