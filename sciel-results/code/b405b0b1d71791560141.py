"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_DEADLINE = time.time() + 45.0
_N = P['n']
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)


def _best_subset():
    best_key = (-1, -1)
    best_set = frozenset(range(P['minimum_vertices']))
    for k in range(P['minimum_vertices'], _N + 1):
        for combo in itertools.combinations(range(_N), k):
            cfro = frozenset(combo)
            degs = set()
            for v in cfro:
                d = 0
                for u in _ADJ[v]:
                    if u in cfro:
                        d += 1
                degs.add(d)
            key = (len(degs), k)
            if key > best_key:
                best_key = key
                best_set = cfro
    return best_set


def construct():
    best = _best_subset()
    return [int(v) for v in sorted(best)]
# EVOLVE-BLOCK-END
