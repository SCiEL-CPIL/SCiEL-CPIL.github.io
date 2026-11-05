import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute Clebsch graph edges (4-chromatic, triangle-free)
    clebsch_edges_list = []
    clebsch_set = set()
    for u in range(n):
        for bit_idx in range(4):
            v = u ^ (1 << bit_idx)
            if u < v:
                clebsch_edges_list.append((u, v))
                clebsch_set.add((u, v))
        v = u ^ 15
        if u < v:
            clebsch_edges_list.append((u, v))
            clebsch_set.add((u, v))
            
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    # Fast 3-colorability check using backtracking and bitmasks
    def is_3_colorable(adj):
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = idx
            for c in range(3):
                conflict = False
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)

    # Branch 1: Clebsch-Forced Search
    # Guaranteed to produce valid 4-chromatic graphs.
    # Faster because we skip the 3-colorability check.
    # We allocate 80% of the time to this branch.
    forced_deadline = start_time + (deadline - start_time) * 0.8
    
    while time.time() < forced_deadline:
        adj = [0] * n
        # Initialize with Clebsch edges
        for u, v in clebsch_edges_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Prepare remaining edges
        remaining_edges = [e for e in all_edges if e not in clebsch_set]
        random.shuffle(remaining_edges)
        
        edge_count = len(clebsch_edges_list)
        
        for u, v in remaining_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = get_edges_from_adj(adj)

    # Branch 2: Random Greedy Search
    # Explores the full space, but requires 3-colorability check.
    while time.time() < deadline:
        adj = [0] * n
        edge_count = 0
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        if not is_3_colorable(adj):
            if edge_count > best_count:
                best_count = edge_count
                best_edges = get_edges_from_adj(adj)

    # Fallback
    if best_count == 0:
        best_edges = [[u, v] for u, v in clebsch_edges_list]

    return best_edges
