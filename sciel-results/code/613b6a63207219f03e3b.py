"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import collections

def _moore_bound(k, g):
    """Compute the Moore bound (lower bound) for a k-regular graph of girth g."""
    if k <= 2:
        return g if g % 2 == 0 else 2 * g - 1
    if g % 2 == 0:
        half = g // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 2 * k * s
    else:
        half = (g - 1) // 2
        s = 0
        term = 1
        for i in range(half + 1):
            s += term
            term *= (k - 1)
        return 1 + k * s

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    (LUW) family. For k=5, g=9, produces a 5-regular bipartite graph with girth 10.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    q = k  # For k=5, q=5 (prime)
    
    # Choose d so that girth >= g
    # For odd d: girth >= d + 5
    # For even d: girth >= d + 4
    d = None
    for candidate_d in range(3, 20):
        if candidate_d % 2 == 1:
            if candidate_d + 5 >= g:
                d = candidate_d
                break
        else:
            if candidate_d + 4 >= g:
                d = candidate_d
                break
    
    if d is None:
        d = g - 4
    if d < 3:
        d = 3
    
    # Finite-field helpers
    if q == 4:
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q
    
    # Recurrence rules for LUW construction
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]
    
    # Precompute rule arrays for speed
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c] = rules[c][0]
        rule_b[c] = rules[c][1]
    
    # BFS over the incidence graph
    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    
    idx = 0
    while idx < len(queue):
        if time.time() > DEADLINE:
            break
        vkey = queue[idx]
        idx += 1
        side, coords = vkey
        vidx = label[vkey]
        
        for first_val in range(q):
            # Build neighbour coordinates
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            if side == 0:
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
            else:
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    
    n = len(queue)
    return n, edge_list
# EVOLVE-BLOCK-END
