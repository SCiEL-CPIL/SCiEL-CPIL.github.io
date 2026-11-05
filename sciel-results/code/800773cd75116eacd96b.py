import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    best_edges = []
    best_count = 0
    
    adj = [0] * n
    
    def is_3_colorable():
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if (adj[i] >> j) & 1:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
                    
        colors = [-1] * n
        order = sorted(range(n), key=lambda x: -len(adj_list[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            for nb in adj_list[v]:
                if colors[nb] != -1:
                    used |= (1 << colors[nb])
            available = ~used & 7
            while available:
                c = (available & -available).bit_length() - 1
                available &= available - 1
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        return backtrack(0)
    
    def search(i, current_edges):
        nonlocal best_edges, best_count
        if time.time() > deadline:
            return
        
        if i == n:
            if current_edges > best_count:
                if not is_3_colorable():
                    best_count = current_edges
                    best_edges = []
                    for u in range(n):
                        for v in range(u):
                            if (adj[u] >> v) & 1:
                                best_edges.append([v, u])
            return
        
        max_possible = sum(k for k in range(i, n))
        if current_edges + max_possible <= best_count:
            return
            
        # Generate subsets of 0..i-1
        # Sort by bit count descending
        subsets = list(range(1 << i))
        subsets.sort(key=lambda x: -x.bit_count())
        
        for mask in subsets:
            # Check independence
            m = mask
            valid = True
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if adj[u] & mask:
                    valid = False
                    break
                m -= lsb
            
            if not valid:
                continue
                
            adj[i] = mask
            new_edges = current_edges + mask.bit_count()
            
            search(i + 1, new_edges)
            
            adj[i] = 0
    
    search(0, 0)
    
    if not best_edges:
        # Fallback: Return a known non-3-colorable graph
        # Grötzsch graph (11 vertices) + 5 isolated
        # Edges of Grötzsch:
        # 0-1, 1-2, 2-3, 3-4, 4-0 (C5)
        # 5-1, 5-2, 6-2, 6-3, 7-3, 7-4, 8-4, 8-0, 9-0, 9-1
        # 10-5, 10-6, 10-7, 10-8, 10-9
        # Total 20 edges.
        # But we want max edges.
        # If search failed, we return this.
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
            [5, 1], [5, 2], [6, 2], [6, 3], [7, 3], [7, 4], [8, 4], [8, 0], [9, 0], [9, 1],
            [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
        ]
        return edges
        
    return best_edges
