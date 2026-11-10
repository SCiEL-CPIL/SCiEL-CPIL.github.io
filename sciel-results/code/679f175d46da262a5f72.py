import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
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

def is_not_3_colorable_exact(adj, n=16):
    """Exact 3-colorability check using backtracking with propagation."""
    colors = [-1] * n
    uncolored = (1 << n) - 1
    
    # available_colors[v] is a bitmask of available colors for v
    # Initially 111 for all
    avail = [7] * n
    
    def propagate():
        """Unit propagation. Returns False if contradiction."""
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if colors[v] != -1:
                    continue
                # Check if avail[v] has only 1 bit
                if avail[v] & (avail[v] - 1) == 0:
                    # Only one color available
                    c = avail[v].bit_length() - 1
                    colors[v] = c
                    uncolored &= ~(1 << v)
                    # Update neighbors
                    nb_mask = adj[v]
                    mm = nb_mask
                    while mm:
                        lsb = mm & -mm
                        nb = lsb.bit_length() - 1
                        mm ^= lsb
                        if colors[nb] == -1:
                            if avail[nb] & (1 << c):
                                avail[nb] &= ~(1 << c)
                                if avail[nb] == 0:
                                    return False
                                changed = True
                    changed = True
        return True

    def dfs():
        nonlocal uncolored
        if uncolored == 0:
            return False
        
        # Pick vertex with fewest available colors (MRV)
        best_v = -1
        min_avail = 4
        m = uncolored
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            cnt = avail[v].bit_count()
            if cnt < min_avail:
                min_avail = cnt
                best_v = v
        
        v = best_v
        # Try each available color
        av = avail[v]
        while av:
            lsb = av & -av
            c = lsb.bit_length() - 1
            av ^= lsb
            
            colors[v] = c
            uncolored &= ~(1 << v)
            
            # Save state
            old_avail = list(avail)
            
            # Propagate
            if not propagate():
                # Undo
                uncolored |= 1 << v
                colors[v] = -1
                avail = old_avail
                continue
            
            if not dfs():
                return False
            
            # Undo
            uncolored |= 1 << v
            colors[v] = -1
            avail = old_avail
        
        return True

    if not propagate():
        return True # Contradiction means not colorable? No, if propagation fails, it means no coloring.
    # Wait, if propagate returns False, it means a vertex has 0 available colors.
    # So the graph is not 3-colorable.
    
    return dfs()

def is_not_3_colorable_fast(adj, n=16):
    return is_not_3_colorable_exact(adj, n)

def build_groetzsch_edges_case3():
    # Case 3: a=2, b=1, c=1
    # O1 (0-4) size 2. O2 (5-9) size 1. O3 (10) size 1.
    # Vertices:
    # 0,1 -> O1[0]
    # 2,3 -> O1[1]
    # 4,5 -> O1[2]
    # 6,7 -> O1[3]
    # 8,9 -> O1[4]
    # 10 -> O2[0]
    # 11 -> O2[1]
    # 12 -> O2[2]
    # 13 -> O2[3]
    # 14 -> O2[4]
    # 15 -> O3[0]
    
    edges = []
    # Helper to add edges between sets
    def add_edges(set1, set2):
        for u in set1:
            for v in set2:
                edges.append([u, v])
                
    O1 = [[0,1], [2,3], [4,5], [6,7], [8,9]]
    O2 = [[10], [11], [12], [13], [14]]
    O3 = [[15]]
    
    # Cycle in O1
    for i in range(5):
        j = (i + 1) % 5
        add_edges(O1[i], O1[j])
        
    # O1 to O2
    # Grötzsch structure:
    # O2[i] connected to O1[i] and O1[i+1] (indices mod 5)
    for i in range(5):
        add_edges(O2[i], O1[i])
        add_edges(O2[i], O1[(i+1)%5])
        
    # O2 to O3
    for i in range(5):
        add_edges(O2[i], O3[0])
        
    return edges

def build_groetzsch_edges_case1():
    # Case 1: a=1, b=1, c=6
    # O1 (0-4) size 1. O2 (5-9) size 1. O3 (10) size 6.
    # Vertices:
    # 0 -> O1[0]
    # 1 -> O1[1]
    # 2 -> O1[2]
    # 3 -> O1[3]
    # 4 -> O1[4]
    # 5 -> O2[0]
    # 6 -> O2[1]
    # 7 -> O2[2]
    # 8 -> O2[3]
    # 9 -> O2[4]
    # 10..15 -> O3[0]
    
    edges = []
    def add_edges(set1, set2):
        for u in set1:
            for v in set2:
                edges.append([u, v])
                
    O1 = [[0], [1], [2], [3], [4]]
    O2 = [[5], [6], [7], [8], [9]]
    O3 = [[10, 11, 12, 13, 14, 15]]
    
    # Cycle in O1
    for i in range(5):
        j = (i + 1) % 5
        add_edges(O1[i], O1[j])
        
    # O1 to O2
    for i in range(5):
        add_edges(O2[i], O1[i])
        add_edges(O2[i], O1[(i+1)%5])
        
    # O2 to O3
    for i in range(5):
        add_edges(O2[i], O3[0])
        
    return edges

def local_search_add_edges(adj, edges, n=16, deadline=None):
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for u in range(n):
            if deadline and time.time() > deadline:
                break
            for v in range(u+1, n):
                if deadline and time.time() > deadline:
                    break
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append([u, v])
                
                if not is_not_3_colorable_fast(adj, n):
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    edges.pop()
                else:
                    improved = True
    return edges

def local_search_swaps(adj, edges, n=16, deadline=None):
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(len(edges) - 1, -1, -1):
            if deadline and time.time() > deadline:
                break
            u, v = edges[idx]
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            edges.pop(idx)
            
            found = False
            candidates = []
            for a in range(n):
                for b in range(a+1, n):
                    if adj[a] & (1 << b):
                        continue
                    if adj[a] & adj[b]:
                        continue
                    score = (n - 1 - adj[a].bit_count()) + (n - 1 - adj[b].bit_count())
                    candidates.append((score, a, b))
            
            candidates.sort(reverse=True)
            
            for score, a, b in candidates:
                if deadline and time.time() > deadline:
                    break
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                edges.append([a, b])
                
                if is_not_3_colorable_fast(adj, n):
                    found = True
                    improved = True
                    break
                else:
                    adj[a] ^= 1 << b
                    adj[b] ^= 1 << a
                    edges.pop()
            
            if not found:
                edges.append([u, v])
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    deadline = START_TIME + TIME_LIMIT
    
    best_edges = []
    best_count = 0
    
    # Try Case 3
    edges3 = build_groetzsch_edges_case3()
    adj3 = build_adj(edges3, n)
    if is_triangle_free_bitmask(adj3, n) and is_not_3_colorable_fast(adj3, n):
        if len(edges3) > best_count:
            best_count = len(edges3)
            best_edges = list(edges3)
        # Local search
        cur = local_search_add_edges(adj3, list(edges3), n, deadline)
        if cur and len(cur) > best_count:
            best_count = len(cur)
            best_edges = cur
        if time.time() < deadline:
            cur = local_search_swaps(adj3, list(best_edges), n, deadline)
            if cur and len(cur) > best_count:
                best_count = len(cur)
                best_edges = cur

    # Try Case 1
    if time.time() < deadline:
        edges1 = build_groetzsch_edges_case1()
        adj1 = build_adj(edges1, n)
        if is_triangle_free_bitmask(adj1, n) and is_not_3_colorable_fast(adj1, n):
            if len(edges1) > best_count:
                best_count = len(edges1)
                best_edges = list(edges1)
            cur = local_search_add_edges(adj1, list(edges1), n, deadline)
            if cur and len(cur) > best_count:
                best_count = len(cur)
                best_edges = cur
            if time.time() < deadline:
                cur = local_search_swaps(adj1, list(best_edges), n, deadline)
                if cur and len(cur) > best_count:
                    best_count = len(cur)
                    best_edges = cur

    if not best_edges:
        best_edges = build_groetzsch_edges_case3()
        
    return best_edges
