# EVOLVE-BLOCK-START
import time

_P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

_N = _P['n']
_EDGES = _P['edges']

# adjacency bitmasks
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

def _is_clique_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != rest:
            return False
        mask = rest
    return True

def _is_indep_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != 0:
            return False
        mask = rest
    return True

# precompute clique/indep for all masks
_CLIQ = [False] * (1 << _N)
_INDEP = [False] * (1 << _N)
for _m in range(1 << _N):
    if _m == 0:
        _CLIQ[_m] = True
        _INDEP[_m] = True
        continue
    _CLIQ[_m] = _is_clique_mask(_m)
    _INDEP[_m] = _is_indep_mask(_m)

# Precompute: for each mask, can it be covered by k parts where each part is clique or indep?
_VALID_PART = [False] * (1 << _N)
for _m in range(1 << _N):
    if _CLIQ[_m] or _INDEP[_m]:
        _VALID_PART[_m] = True

# Precompute minimum parts to cover each subset (DP over subsets)
_MIN_PARTS = [0] * (1 << _N)
for _m in range(1, 1 << _N):
    _MIN_PARTS[_m] = 10**9

for _m in range(1, 1 << _N):
    best = 10**9
    lowbit = _m & (-_m)
    s = _m
    while s:
        if (s & lowbit) and _VALID_PART[s]:
            rest = _m ^ s
            if _MIN_PARTS[rest] + 1 < best:
                best = _MIN_PARTS[rest] + 1
        s = (s - 1) & _m
    _MIN_PARTS[_m] = best

_DEADLINE = None
_best = None
_best_len = 10**9

def _greedy_incumbent():
    best_parts = None
    best_len = 10**9
    degs = [bin(_ADJ[v]).count('1') for v in range(_N)]
    orders = [
        list(range(_N)),
        sorted(range(_N), key=lambda v: -degs[v]),
        sorted(range(_N), key=lambda v: degs[v]),
        sorted(range(_N), key=lambda v: -degs[v]),
    ]
    for order in orders:
        parts = []  # list of (mask, type) type in {1:clique, 0:indep, 2:free-singleton}
        for v in order:
            bit = 1 << v
            placed = False
            for i, (mask, typ) in enumerate(parts):
                nm = mask | bit
                if typ == 2:
                    if _CLIQ[nm]:
                        parts[i] = (nm, 1)
                    elif _INDEP[nm]:
                        parts[i] = (nm, 0)
                    else:
                        continue
                    placed = True
                    break
                elif typ == 1:
                    if _CLIQ[nm]:
                        parts[i] = (nm, 1)
                        placed = True
                        break
                else:
                    if _INDEP[nm]:
                        parts[i] = (nm, 0)
                        placed = True
                        break
            if not placed:
                parts.append((bit, 2))
        if len(parts) < best_len:
            best_len = len(parts)
            best_parts = []
            for mask, typ in parts:
                vs = [x for x in range(_N) if (mask >> x) & 1]
                best_parts.append(vs)
    return best_parts

def _dfs(remaining_mask, parts, depth):
    global _best, _best_len
    if remaining_mask == 0:
        if depth < _best_len:
            _best_len = depth
            _best = [[x for x in range(_N) if (mask >> x) & 1] for (mask, typ) in parts]
        return
    if depth >= _best_len:
        return
    if time.time() > _DEADLINE:
        return
    # Lower bound: _MIN_PARTS[remaining_mask]
    lb = _MIN_PARTS[remaining_mask]
    if depth + lb >= _best_len:
        return
    # Choose vertex with fewest options
    rem_list = [x for x in range(_N) if (remaining_mask >> x) & 1]
    
    def options_count(v):
        bit = 1 << v
        c = 0
        for (mask, typ) in parts:
            nm = mask | bit
            if typ == 2:
                if _CLIQ[nm] or _INDEP[nm]:
                    c += 1
            elif typ == 1:
                if _CLIQ[nm]:
                    c += 1
            else:
                if _INDEP[nm]:
                    c += 1
        return c
    
    v = min(rem_list, key=lambda x: (options_count(x), -bin(_ADJ[x]).count('1')))
    vbit = 1 << v
    new_rem = remaining_mask ^ vbit
    
    # Try placing into existing parts
    placed_in_existing = False
    for i, (mask, typ) in enumerate(parts):
        nm = mask | vbit
        newtyp = None
        if typ == 2:
            if _CLIQ[nm]:
                newtyp = 1
            elif _INDEP[nm]:
                newtyp = 0
            else:
                continue
        elif typ == 1:
            if _CLIQ[nm]:
                newtyp = 1
            else:
                continue
        else:
            if _INDEP[nm]:
                newtyp = 0
            else:
                continue
        parts[i] = (nm, newtyp)
        _dfs(new_rem, parts, depth + 1)
        parts[i] = (mask, typ)
        placed_in_existing = True
        if _best_len <= depth + 1:
            return
    
    # Try opening a new part - only if we didn't place in an existing free singleton
    # This breaks symmetry: don't open a new singleton if there's already a free one we could have used
    has_free = any(typ == 2 for (_, typ) in parts)
    if not has_free:
        parts.append((vbit, 2))
        _dfs(new_rem, parts, depth + 1)
        parts.pop()

def construct():
    global _best, _best_len, _DEADLINE
    _DEADLINE = time.time() + 80.0
    # Initial incumbent via greedy
    gp = _greedy_incumbent()
    _best = gp
    _best_len = len(gp)
    # Exact search
    _dfs((1 << _N) - 1, [], 0)
    if _best is None:
        _best = [list(range(_N))]
    out = []
    for part in _best:
        out.append([int(x) for x in part])
    return out
# EVOLVE-BLOCK-END
