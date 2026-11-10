"""MWIS on 3-AP conflict graph with optimized branch-and-bound."""
import time

N = 256
WEIGHT = [1.0 / (i + 1) for i in range(N)]

def build_conflicts():
    conf = [0] * N
    for p in range(N):
        x = p + 1
        for q in range(p + 1, N):
            y = q + 1
            d = y - x
            hit = False
            if x - d >= 1:
                hit = True
            elif y + d <= 256:
                hit = True
            else:
                if (x + y) % 2 == 0:
                    m = (x + y) // 2
                    if 1 <= m <= 256 and m != x and m != y:
                        hit = True
            if hit:
                conf[p] |= (1 << q)
                conf[q] |= (1 << p)
    return conf

CONF = build_conflicts()

_start_time = None
_best_set = None
_best_val = 0.0
_time_limit = 50.0

def _time_left():
    return _start_time is None or (time.time() - _start_time) < _time_limit

def _greedy_incumbent(cand_mask):
    """Find a good independent set within cand_mask to update best_val."""
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    verts.sort(key=lambda i: -WEIGHT[i])
    chosen = []
    avail = cand_mask
    for v in verts:
        if (avail >> v) & 1:
            chosen.append(v)
            avail &= ~CONF[v]
            avail &= ~(1 << v)
    return sum(WEIGHT[i] for i in chosen), chosen

def _color_ub(cand_mask):
    """
    Upper bound using greedy coloring of the conflict graph induced by cand_mask.
    """
    if cand_mask == 0:
        return 0.0
    
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    verts.sort(key=lambda i: -WEIGHT[i])
    
    color_of = [-1] * N
    col_max = {}
    
    for v in verts:
        used_colors = set()
        cm = CONF[v] & cand_mask
        while cm:
            nb = (cm & -cm).bit_length() - 1
            cm &= cm - 1
            if color_of[nb] != -1:
                used_colors.add(color_of[nb])
        
        c = 0
        while c in used_colors:
            c += 1
        color_of[v] = c
        
        if c not in col_max or WEIGHT[v] > col_max[c]:
            col_max[c] = WEIGHT[v]
            
    return sum(col_max.values())

def _search(cand_mask, cur_val, chosen):
    global _best_set, _best_val
    
    if not _time_left():
        return
        
    if cand_mask == 0:
        if cur_val > _best_val:
            _best_val = cur_val
            _best_set = list(chosen)
        return

    # Update incumbent with a greedy solution from the current state
    g_val, g_set = _greedy_incumbent(cand_mask)
    if g_val > _best_val:
        _best_val = g_val
        _best_set = list(chosen) + g_set

    # Prune using coloring bound
    ub = _color_ub(cand_mask)
    if cur_val + ub <= _best_val + 1e-12:
        return

    # Branching: Pick vertex with max weight in cand_mask
    max_v = -1
    max_w = -1.0
    cm = cand_mask
    while cm:
        v = (cm & -cm).bit_length() - 1
        cm &= cm - 1
        if WEIGHT[v] > max_w:
            max_w = WEIGHT[v]
            max_v = v
            
    if max_v == -1:
        return

    # Branch 1: Include max_v
    new_cand = cand_mask & ~CONF[max_v] & ~(1 << max_v)
    _search(new_cand, cur_val + WEIGHT[max_v], chosen + [max_v])
    
    if not _time_left():
        return

    # Branch 2: Exclude max_v
    _search(cand_mask & ~(1 << max_v), cur_val, chosen)

def construct():
    global _start_time, _best_set, _best_val
    _start_time = time.time()
    _best_val = 0.0
    _best_set = None
    full = (1 << N) - 1
    
    # Initial greedy solution to set a strong lower bound
    g_val, g_set = _greedy_incumbent(full)
    _best_val = g_val
    _best_set = g_set
    
    _search(full, 0.0, [])
    
    if _best_set is None:
        _best_set = [0]
        
    return [i + 1 for i in _best_set]
