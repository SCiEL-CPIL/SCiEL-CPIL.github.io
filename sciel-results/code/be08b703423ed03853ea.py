import time
import random
from itertools import combinations

_N = 16
_TIME_LIMIT = 115.0

def build_adj(edges, n=_N):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free(adj, n=_N):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable(adj, n=_N):
    """Exact backtracking 3-colouring. Returns True iff the graph is NOT 3-colourable."""
    memo = set()

    def dfs(uncolored_mask, colors):
        if uncolored_mask == 0:
            return False
        key = (uncolored_mask, tuple(colors))
        if key in memo:
            return True
        # pick the vertex with the most already-coloured neighbours
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb = adj[v]
            while nb:
                lsb2 = nb & -nb
                w = lsb2.bit_length() - 1
                nb ^= lsb2
                if colors[w] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v
        forbidden = 0
        nb = adj[v]
        while nb:
            lsb2 = nb & -nb
            w = lsb2.bit_length() - 1
            nb ^= lsb2
            if colors[w] != -1:
                forbidden |= 1 << colors[w]
        ok = True
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v), colors):
                    ok = False
                    break
                colors[v] = -1
        if ok:
            memo.add(key)
        return ok

    return dfs((1 << n) - 1, [-1] * n)

def build_clebsch_edges():
    edges = []
    for a in range(16):
        i1, j1 = a // 4, a % 4
        for b in range(a + 1, 16):
            i2, j2 = b // 4, b % 4
            if i1 != i2 and j1 != j2:
                edges.append([a, b])
    return edges

def _polish_graph(edges, start_time, deadline):
    n = _N
    adj = build_adj(edges, n)
    cur_edges = list(edges)
    
    if not is_triangle_free(adj, n):
        return None, 0
    if not is_not_3_colorable(adj, n):
        return None, 0

    best_edges = list(cur_edges)
    best_count = len(cur_edges)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any single edge
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        if candidates:
            # Add the first valid candidate (or random)
            # To be deterministic and fast, just pick the first
            u, v = candidates[0]
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            cur_edges.append([u, v])
            
            if is_not_3_colorable(adj, n):
                best_count = len(cur_edges)
                best_edges = list(cur_edges)
                improved = True
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                cur_edges.pop()
        
        if improved:
            continue
        
        # 2. Try to swap: remove 1 edge, add 2 edges (net +1)
        # This is more expensive. Iterate through all edges to remove.
        swap_found = False
        edges_to_try = list(cur_edges)
        random.shuffle(edges_to_try)
        
        for eu, ev in edges_to_try:
            if time.time() > deadline:
                break
            
            # Remove edge (eu, ev)
            adj[eu] ^= 1 << ev
            adj[ev] ^= 1 << eu
            cur_edges.remove([eu, ev])
            
            # Find all possible additions
            add_candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            add_candidates.append((u, v))
            
            if len(add_candidates) >= 2:
                # Try adding 2 edges
                # We need to check if adding 2 specific edges keeps it non-3-colorable
                # To save time, just try the first 2
                (u1, v1), (u2, v2) = add_candidates[0], add_candidates[1]
                
                # Add first
                adj[u1] |= 1 << v1
                adj[v1] |= 1 << u1
                # Add second
                adj[u2] |= 1 << v2
                adj[v2] |= 1 << u2
                
                if is_not_3_colorable(adj, n):
                    cur_edges.append([u1, v1])
                    cur_edges.append([u2, v2])
                    best_count = len(cur_edges)
                    best_edges = list(cur_edges)
                    swap_found = True
                    break
                else:
                    # Revert additions
                    adj[u1] ^= 1 << v1
                    adj[v1] ^= 1 << u1
                    adj[u2] ^= 1 << v2
                    adj[v2] ^= 1 << u2
            
            # Revert removal
            adj[eu] |= 1 << ev
            adj[ev] |= 1 << eu
            cur_edges.append([eu, ev])
            
            if swap_found:
                break
        
        if not improved and not swap_found:
            break

    return best_edges, best_count

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_edges = None
    best_count = 0

    # 1. Try Clebsch polish
    clebsch_edges = build_clebsch_edges()
    c_edges, c_count = _polish_graph(clebsch_edges, start_time, deadline)
    if c_edges and c_count > best_count:
        best_edges = c_edges
        best_count = c_count

    if best_edges is None:
        return build_clebsch_edges()
        
    return best_edges
