import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 31
    q = 5

    # Build polarity graph of PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    assert len(points) == n
    
    # Build adjacency - polarity graph is C4-free with 90 edges
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    # Polarity graph of PG(2,5) should have 90 edges (C4-free)
    # We need exactly 91 edges, so add 1 edge.
    
    def count_c4s(adj_sets):
        total = 0
        for u in range(n):
            nu = adj_sets[u]
            for v in range(u+1, n):
                nv = adj_sets[v]
                common = len(nu & nv)
                total += common * (common - 1) // 2
        return total // 2
    
    if m == 90:
        # Need to add 1 edge. Find the non-edge that creates fewest C4s.
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        best_cost = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    # Cost = number of C4s created by adding (i,j)
                    # A C4 through (i,j) is i-a-b-j-i where a~i, b~j, a~b
                    nu = adj[i]
                    nv = adj[j]
                    count = 0
                    for a in nu:
                        for b in nv:
                            if a != b and b in adj[a]:
                                count += 1
                    if count < best_cost:
                        best_cost = count
                        best_edge = [i, j]
        
        if best_edge is not None:
            u, v = best_edge
            edges.append(best_edge)
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
    
    elif m > 91:
        # Remove (m - 91) edges optimally
        k = m - 91
        if k == 2:
            best_c4s = count_c4s(adj)
            best_remove = None
            
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, m):
                    new_adj = [set(s) for s in adj]
                    u1, v1 = edges[i]
                    u2, v2 = edges[j]
                    new_adj[u1].discard(v1)
                    new_adj[v1].discard(u1)
                    new_adj[u2].discard(v2)
                    new_adj[v2].discard(u2)
                    
                    c4s = count_c4s(new_adj)
                    if c4s < best_c4s:
                        best_c4s = c4s
                        best_remove = (i, j)
                    
                    if time.time() > deadline - 5:
                        break
                
                if time.time() > deadline - 5:
                    break
            
            if best_remove is not None:
                i, j = best_remove
                edges_to_remove = {tuple(sorted(edges[i])), tuple(sorted(edges[j]))}
                edges = [e for e in edges if tuple(sorted(e)) not in edges_to_remove]
    
    elif m < 91:
        # Add edges greedily
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        while len(edges) < 91:
            best_cost = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        nu = adj[i]
                        nv = adj[j]
                        count = 0
                        for a in nu:
                            for b in nv:
                                if a != b and b in adj[a]:
                                    count += 1
                        if count < best_cost:
                            best_cost = count
                            best_edge = [i, j]
            
            if best_edge is not None:
                edges.append(best_edge)
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v) if u < v else (v, u))
            else:
                break
    
    # Local search: try swapping edges to reduce C4 count
    # An edge swap: remove edge e1, add non-edge e2
    current_c4s = count_c4s(adj)
    
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
    
    # Precompute edge index map
    edge_to_idx = {}
    for idx, e in enumerate(edges):
        key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        edge_to_idx[key] = idx
    
    # Build list of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # Local search with multiple restarts
    best_edges = edges[:]
    best_c4s_global = current_c4s
    
    # Try several random initial states via different extra edge choices
    # First, let's do local search on the current state
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for idx, e in enumerate(edges):
            if time.time() > deadline - 20:
                break
            u, v = e
            # Try removing this edge and adding a non-edge
            # For efficiency, only consider non-edges that could improve things
            # A non-edge (a,b) is good if it creates fewer C4s than the edge (u,v) destroys
            
            # Count C4s that use edge (u,v)
            c4s_using_uv = 0
            for x in adj[v]:
                if x == u: continue
                for y in adj[u]:
                    if y == v or y == x: continue
                    if y in adj[x]:
                        c4s_using_uv += 1
            
            # Try some candidate non-edges
            # Prioritize non-edges with low "cost" (few C4s they would create)
            # But we need to be careful: the graph changes when we remove (u,v)
            
            # For now, just try a few random non-edges
            candidates = random.sample(non_edges, min(20, len(non_edges)))
            for (a, b) in candidates:
                # Skip if (a,b) is adjacent to u or v in a way that doesn't help
                # Just try it
                new_adj = [set(s) for s in adj]
                new_adj[u].discard(v)
                new_adj[v].discard(u)
                new_adj[a].add(b)
                new_adj[b].add(a)
                
                new_c4s = count_c4s(new_adj)
                if new_c4s < current_c4s:
                    # Accept the swap
                    edges[idx] = [a, b]
                    adj = new_adj
                    edge_set.remove((u, v) if u < v else (v, u))
                    edge_set.add((a, b) if a < b else (b, a))
                    current_c4s = new_c4s
                    improved = True
                    if current_c4s < best_c4s_global:
                        best_c4s_global = current_c4s
                        best_edges = [e[:] for e in edges]
                    break  # Restart the loop with the new state
        
        # After each pass, rebuild non_edges and edge_to_idx
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
    
    # Final polish: try all single-edge swaps more exhaustively
    if time.time() < deadline - 10:
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(len(edges)):
                if time.time() > deadline - 10:
                    break
                u, v = edges[idx]
                for (a, b) in non_edges:
                    # Skip if swap doesn't change the edge set meaningfully
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    new_adj = [set(s) for s in adj]
                    new_adj[u].discard(v)
                    new_adj[v].discard(u)
                    new_adj[a].add(b)
                    new_adj[b].add(a)
                    
                    new_c4s = count_c4s(new_adj)
                    if new_c4s < current_c4s:
                        edges[idx] = [a, b]
                        adj = new_adj
                        edge_set.remove((u, v) if u < v else (v, u))
                        edge_set.add((a, b) if a < b else (b, a))
                        current_c4s = new_c4s
                        improved = True
                        if current_c4s < best_c4s_global:
                            best_c4s_global = current_c4s
                            best_edges = [e[:] for e in edges]
                        break
                if improved:
                    break
            
            if improved:
                # Rebuild non_edges
                edge_set = set()
                for e in edges:
                    edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) not in edge_set:
                            non_edges.append((i, j))
    
    # Use the best edges found
    edges = best_edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    return edges
