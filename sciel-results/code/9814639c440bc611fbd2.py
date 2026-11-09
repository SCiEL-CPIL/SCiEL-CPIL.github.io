"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

# Precompute the 60 undirected Hamiltonian cycles on vertices 0..5
_CYCLES = []
_seen = set()
for _perm in permutations(range(1, 6)):
    _cycle = (0,) + _perm
    _edges = frozenset(frozenset((_cycle[k], _cycle[(k + 1) % 6])) for k in range(6))
    if _edges not in _seen:
        _seen.add(_edges)
        _CYCLES.append(_cycle)

def _has_c6_fast(adj, n):
    """Check if boolean adjacency matrix adj (n x n) contains a C6."""
    if n < 6:
        return False
    for subset in combinations(range(n), 6):
        for cycle in _CYCLES:
            ok = True
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k + 1) % 6]]
                if not adj[u, w]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _verify(colors, n):
    """Verify that the coloring has no C6 in any color class."""
    if n < 6:
        return True
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = True
            adj[c][j, i] = True
            idx += 1
    for c in range(4):
        if _has_c6_fast(adj[c], n):
            return False
    return True

def _edge_index(i, j, n):
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj_matrices(colors, n):
    adjs = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i, j] = True
            adjs[c][j, i] = True
            idx += 1
    return adjs

def _random_coloring(n):
    m = n * (n - 1) // 2
    return np.random.randint(0, 4, size=m).tolist()

def _local_search(colors, n, deadline):
    """Local search: iteratively find C6 violations and flip edges to break them."""
    for _ in range(50000):
        if time.time() > deadline:
            break
        adjs = _build_adj_matrices(colors, n)
        found = False
        for c in range(4):
            if _has_c6_fast(adjs[c], n):
                for subset in combinations(range(n), 6):
                    for cycle in _CYCLES:
                        ok = True
                        for k in range(6):
                            u = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if not adjs[c][u, w]:
                                ok = False
                                break
                        if ok:
                            k = np.random.randint(6)
                            v = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if v > w:
                                v, w = w, v
                            idx = _edge_index(v, w, n)
                            old_color = colors[idx]
                            new_color = np.random.randint(0, 4)
                            while new_color == old_color:
                                new_color = np.random.randint(0, 4)
                            colors[idx] = new_color
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if not found:
            if _verify(colors, n):
                return colors
            break
    return colors

def _insert_vertex(colors, n_current, n_target, deadline):
    """
    Try to extend a valid coloring of K_{n_current} to K_{n_target}
    by inserting one vertex at a time.
    """
    cur_colors = list(colors)
    cur_n = n_current
    
    while cur_n < n_target:
        if time.time() > deadline:
            return None
        
        new_n = cur_n + 1
        adjs = _build_adj_matrices(cur_colors, cur_n)
        
        # Precompute path3[c][i][j]
        path3 = [[[False] * cur_n for _ in range(cur_n)] for _ in range(4)]
        for c in range(4):
            adj_c = adjs[c]
            for i in range(cur_n):
                for j in range(cur_n):
                    if i == j:
                        continue
                    found = False
                    for a in range(cur_n):
                        if a == i or a == j:
                            continue
                        if not adj_c[i, a]:
                            continue
                        for b in range(cur_n):
                            if b == i or b == j or b == a:
                                continue
                            if adj_c[a, b] and adj_c[b, j]:
                                found = True
                                break
                        if found:
                            break
                    path3[c][i][j] = found
        
        assign = [-1] * cur_n
        
        def _bt(k):
            if time.time() > deadline:
                return False
            if k == cur_n:
                return True
            for c in range(4):
                valid = True
                for j in range(k):
                    if assign[j] == c and path3[c][j][k]:
                        valid = False
                        break
                if not valid:
                    continue
                assign[k] = c
                if _bt(k + 1):
                    return True
                assign[k] = -1
            return False
        
        if _bt(0):
            new_colors = list(cur_colors)
            for i in range(cur_n):
                new_colors.append(assign[i])
            cur_colors = new_colors
            cur_n = new_n
        else:
            return None
    
    return cur_colors

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 1
    best_colors = []
    
    for target_n in range(2, 21):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            if best_n >= target_n - 1:
                base_colors = list(best_colors)
                base_n = best_n
            else:
                base_n = target_n - 1
                base_colors = _random_coloring(base_n)
                base_colors = _local_search(base_colors, base_n, deadline)
                if not _verify(base_colors, base_n):
                    continue
            
            new_colors = _insert_vertex(base_colors, base_n, target_n, deadline)
            if new_colors is not None and _verify(new_colors, target_n):
                best_n = target_n
                best_colors = new_colors
                success = True
                break
        
        if not success:
            for restart in range(30):
                if time.time() > deadline - 5:
                    break
                colors = _random_coloring(target_n)
                colors = _local_search(colors, target_n, deadline)
                if _verify(colors, target_n):
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
