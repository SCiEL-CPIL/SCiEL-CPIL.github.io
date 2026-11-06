import random
import time

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _build_chvatal_edges():
    """Return the edge list of the Chvátal graph on vertices 0..11."""
    # Standard Chvátal graph edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 0],
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [7, 0], [8, 1], [9, 2], [10, 3], [11, 4]
    ]
    # Wait, I need to verify this edge list.
    # Let's use a known valid edge list for Chvátal graph.
    # From Wikipedia/Graph Atlas:
    # Vertices 0-11.
    # 0: 1, 5, 6, 10
    # 1: 0, 2, 6, 11
    # 2: 1, 3, 7, 11
    # 3: 2, 4, 7, 10
    # 4: 3, 5, 8, 10
    # 5: 0, 4, 8, 9
    # 6: 0, 1, 9, 11
    # 7: 2, 3, 9, 10
    # 8: 4, 5, 10, 11
    # 9: 5, 6, 7, 11
    # 10: 0, 3, 7, 8
    # 11: 1, 2, 8, 9
    # Let's construct from this adjacency.
    adj = {
        0: [1, 5, 6, 10],
        1: [0, 2, 6, 11],
        2: [1, 3, 7, 11],
        3: [2, 4, 7, 10],
        4: [3, 5, 8, 10],
        5: [0, 4, 8, 9],
        6: [0, 1, 9, 11],
        7: [2, 3, 9, 10],
        8: [4, 5, 10, 11],
        9: [5, 6, 7, 11],
        10: [0, 3, 7, 8],
        11: [1, 2, 8, 9]
    }
    edges = []
    for u in range(12):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

def _make_bitmask_adj(edge_list, n=16):
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj):
    n = len(adj)
    color = [0] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        best_v = -1
        min_avail = 8
        for v in range(n):
            if color[v] == 0:
                unavail = 0
                m = adj[v]
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    m ^= lb
                    if color[u] != 0:
                        unavail |= (1 << (color[u]-1))
                avail_colors = 7 ^ unavail
                count = bin(avail_colors).count('1')
                if count == 0:
                    return False
                if count < min_avail:
                    min_avail = count
                    best_v = v
                    if count == 1:
                        break
        
        if best_v == -1:
            return True
            
        mask = adj[best_v]
        unavail = 0
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if color[u] != 0:
                unavail |= (1 << (color[u]-1))
        
        avail_colors = 7 ^ unavail
        
        c = 1
        while c <= 3:
            if avail_colors & (1 << (c-1)):
                color[best_v] = c
                if backtrack(idx + 1):
                    return True
                color[best_v] = 0
            c += 1
        return False

    return backtrack(0)

def _greedy_complete_with_core(core_edges, seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    adj = _make_bitmask_adj(core_edges, n)
    edge_set = set()
    for u, v in core_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set]
    rng.shuffle(remaining)
    
    # Greedy addition
    for u, v in remaining:
        if time.time() > deadline:
            break
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                result_edges.append([u, v])
                
    return result_edges, len(result_edges)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    grotzsch = _build_grotzsch_edges()
    chvatal = _build_chvatal_edges()
    
    best_edges = list(grotzsch)
    best_count = len(grotzsch)
    
    cores = [grotzsch, chvatal]
    
    seed = 0
    while time.time() < deadline - 5.0:
        # Alternate cores or just try both in each iteration?
        # Trying both doubles the work per seed.
        # Let's just run one core per seed, alternating.
        core = cores[seed % 2]
        edges, count = _greedy_complete_with_core(core, seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        
    adj = _make_bitmask_adj(best_edges)
    if not _is_triangle_free(adj):
        best_edges = list(grotzsch)
    if _is_3_colorable(adj):
        best_edges = list(grotzsch)
        
    return best_edges
