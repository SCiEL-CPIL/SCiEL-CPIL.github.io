import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin for 150s limit
    
    n = 31
    target_edges = 91
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_graph = None
    best_edge_count = 0
    best_non_edge_codeg1 = None
    
    # We will try to find a C4-free graph with as many edges as possible.
    # If we find one with >= 91 edges, we return 91 of them (0 C4s).
    # If we find one with 90 edges, we find a non-edge with codeg 1 and add it (1 C4).
    # If we find one with < 90 edges, we add edges greedily, accepting C4s, to reach 91.
    
    # To ensure we explore different structures, we shuffle edges in different orders.
    # Number of attempts depends on time.
    attempts = 0
    while time.time() < deadline - 5.0:  # Leave 5s for final processing
        attempts += 1
        random.shuffle(all_edges)
        
        adj = [[False] * n for _ in range(n)]
        edge_count = 0
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is created if there exists w in N(u) and x in N(v) such that (w, x) is an edge.
            # Note: w and x must be distinct from u, v.
            # Also w != x.
            safe = True
            # Optimization: iterate over smaller neighborhood
            nu = [w for w in range(n) if adj[u][w]]
            nv = [x for x in range(n) if adj[v][x]]
            
            # If either is empty, it's safe (no path of length 3)
            if not nu or not nv:
                safe = True
            else:
                # Check all pairs (w, x) with w in nu, x in nv
                # We need adj[w][x] to be True
                for w in nu:
                    if w == v: continue
                    for x in nv:
                        if x == w: continue
                        if adj[w][x]:
                            safe = False
                            break
                    if not safe:
                        break
            
            if safe:
                adj[u][v] = True
                adj[v][u] = True
                edge_count += 1
                current_edges.append((u, v))
                
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    # Save a copy of the adjacency matrix or edge list
                    # For efficiency, we might just save the edge list and rebuild adj if needed later
                    best_graph_edges = current_edges[:]
                    # We also need to know if this graph is C4-free (it is, by construction)
                    
        # After greedy pass, we have a maximal C4-free graph
        if edge_count >= target_edges:
            # We can just return the first 91 edges
            # Since the graph is C4-free, any subset is C4-free
            return [list(e) for e in best_graph_edges[:target_edges]]
            
        elif edge_count == target_edges - 1:
            # We have 90 edges, C4-free.
            # Find a non-edge (u, v) with codeg(u, v) == 1.
            # Rebuild adj for best_graph_edges if it's different from current (it is same structure)
            # Actually, current_edges is the one.
            
            # Find all non-edges
            non_edges = []
            for u, v in all_edges:
                if not adj[u][v]:
                    non_edges.append((u, v))
            
            found_codeg1 = None
            for u, v in non_edges:
                # Calculate codeg
                codeg = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        codeg += 1
                if codeg == 1:
                    found_codeg1 = (u, v)
                    break
            
            if found_codeg1:
                final_edges = current_edges + [list(found_codeg1)]
                return final_edges
            else:
                # If no codeg 1 found, pick any non-edge (will create >= 2 C4s)
                # Or just return the 90 edges? No, must be 91.
                # Pick non-edge with min codeg
                min_codeg = float('inf')
                best_ne = None
                for u, v in non_edges:
                    codeg = 0
                    for w in range(n):
                        if adj[u][w] and adj[v][w]:
                            codeg += 1
                    if codeg < min_codeg:
                        min_codeg = codeg
                        best_ne = (u, v)
                final_edges = current_edges + [list(best_ne)]
                return final_edges

    # If we exit the loop, use the best graph found so far
    if best_graph is None or best_edge_count == 0:
        # Fallback: return lexicographic first 91
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    # Reconstruct best graph
    adj = [[False] * n for _ in range(n)]
    for u, v in best_graph_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    if best_edge_count >= target_edges:
        return [list(e) for e in best_graph_edges[:target_edges]]
        
    # Need to add edges
    current_edges = best_graph_edges[:]
    current_count = best_edge_count
    
    # Generate list of non-edges
    non_edges = []
    for u, v in all_edges:
        if not adj[u][v]:
            non_edges.append((u, v))
            
    # Sort non-edges by codeg to add "cheapest" edges first
    # Codeg in current graph
    def calc_codeg(u, v, adj):
        c = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                c += 1
        return c
        
    # We need to add (target_edges - current_count) edges
    # Greedily pick non-edge with min codeg
    # Note: adding an edge changes codegs for other pairs, but for small additions this is ok.
    # For better accuracy, we should re-evaluate or just pick min.
    
    to_add = target_edges - current_count
    
    # Create a list of (codeg, u, v)
    candidates = []
    for u, v in non_edges:
        c = calc_codeg(u, v, adj)
        candidates.append((c, u, v))
        
    candidates.sort()
    
    added = 0
    for c, u, v in candidates:
        if added >= to_add:
            break
        # Add edge
        current_edges.append([u, v])
        adj[u][v] = True
        adj[v][u] = True
        added += 1
        
    return current_edges
