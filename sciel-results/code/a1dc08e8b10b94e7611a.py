import time

def construct():
    start_time = time.time()
    _P = 1009
    _M = 5

    # Precompute quadratic residues and one root for each
    is_residue = [False] * _P
    root_map = {}

    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_residue[r] = True
            if r not in root_map:
                root_map[r] = x

    # Compute prefix sum of the residue indicator array
    # prefix[i] = number of residues in [0, i-1]
    prefix = [0] * (_P + 1)
    for i in range(_P):
        prefix[i + 1] = prefix[i] + (1 if is_residue[i] else 0)

    # Scan s from 1 to _P - _M (so s+4 <= _P-1 = 1008)
    # Check if s, s+1, s+2, s+3, s+4 are all residues using prefix sums
    # prefix[s+5] - prefix[s] == 5 means all 5 are residues
    for s in range(1, _P - _M + 1):
        if prefix[s + _M] - prefix[s] == _M:
            roots = [root_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No baseline run found')
