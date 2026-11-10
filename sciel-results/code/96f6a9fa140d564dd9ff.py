import itertools
import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110
    
    q3_edges_idx = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    subsets = list(itertools.combinations(range(n), 8))
    
    def find_q3_edges(vertices, adj):
        # vertices: tuple of 8
        # Returns list of 12 edges (as (u,v) with u<v) or None
        # Quick check
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i]][vertices[j]]:
                    ec += 1
        if ec < 12:
            return None
            
        # Backtracking to find mapping
        # mapping: q3_vertex -> index in vertices (0..7)
        for start_v in range(8):
            mapping = {0: start_v}
            def bt(q3_v):
                if q3_v == 8:
                    return True
                for i in range(8):
                    if i not in mapping.values():
                        ok = True
                        for prev_v in range(q3_v):
                            # Check if prev_v and q3_v are connected in Q3
                            # Precomputed set of Q3 edges
                            if (prev_v, q3_v) in q3_edges_set or (q3_v, prev_v) in q3_edges_set:
                                prev_idx = mapping[prev_v]
                                if not adj[vertices[prev_idx]][vertices[i]]:
                                    ok = False
                                    break
                        if ok:
                            mapping[q3_v] = i
                            if bt(q3_v + 1):
                                return True
                            del mapping[q3_v]
                return False
            if bt(1):
                # Found a mapping. Construct the edges.
                edges = []
                for u, v in q3_edges_idx:
                    u_idx = mapping[u]
                    v_idx = mapping[v]
                    u_node = vertices[u_idx]
                    v_node = vertices[v_idx]
                    if u_node > v_node:
                        u_node, v_node = v_node, u_node
                    edges.append((u_node, v_node))
                return edges
        return None

    q3_edges_set = set()
    for u, v in q3_edges_idx:
        q3_edges_set.add((u, v))
        q3_edges_set.add((v, u))
        
    best_edges = 0
    best_adj = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = True
                adj[j][i] = True
                
        while True:
            if time.time() > deadline:
                break
                
            q3s_found = []
            for subset in subsets:
                edges = find_q3_edges(subset, adj)
                if edges:
                    q3s_found.append(edges)
                    
            if not q3s_found:
                break
                
            # Count edge frequencies
            edge_counts = {}
            for edges in q3s_found:
                for e in edges:
                    edge_counts[e] = edge_counts.get(e, 0) + 1
                    
            # Find max
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e = random.choice(candidates)
            u, v = e
            adj[u][v] = False
            adj[v][u] = False
            
        ec = sum(sum(row) for row in adj) // 2
        if ec > best_edges:
            best_edges = ec
            best_adj = [row[:] for row in adj]
            
    res = []
    if best_adj:
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    res.append([i, j])
    else:
        res = [[a, b] for a in range(3) for b in range(3, 16)]
        
    return res
