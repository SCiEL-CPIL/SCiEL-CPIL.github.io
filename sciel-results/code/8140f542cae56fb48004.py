"""Improved K4-free graph construction for n=18."""
import time
import random

# EVOLVE-BLOCK-START

def _is_k4_free(adj):
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        for l in range(k+1, n):
                            if adj[i][l] and adj[j][l] and adj[k][l]:
                                return False
    return True

def _max_triangle_free_size(adj):
    n = len(adj)
    # Use branch and bound to find the maximum triangle-free induced subgraph
    # Triangle-free means no three vertices are mutually adjacent
    
    # Precompute adjacency sets
    adj_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_sets[i].add(j)
                adj_sets[j].add(i)
    
    # We want to find the maximum set S such that no three vertices in S form a triangle
    # Use a greedy + local search approach with verification
    
    best_size = 0
    best_set = set()
    
    # Try to build a triangle-free set greedily with randomization
    for attempt in range(200):
        # Random order
        order = list(range(n))
        random.shuffle(order)
        
        # Greedily add vertices if they don't create a triangle
        current = set()
        for v in order:
            # Check if adding v to current creates a triangle
            can_add = True
            for u in current:
                if adj_sets[u].issubset(current) and v in adj_sets[u]:
                    # Check if there's a w in current adjacent to both u and v
                    # Actually, we need to check if there exists w in current such that
                    # u-w and w-v and u-v are all edges
                    # This is O(|current|) per check, so total O(n^2)
                    for w in current:
                        if w != u and adj_sets[u].__contains__(w) and adj_sets[v].__contains__(w):
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                current.add(v)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current.copy()
    
    return best_size

def _construct_greedy_k4_free():
    """Construct a K4-free graph using greedy edge addition."""
    n = 18
    # Start with empty graph
    adj = [[False]*n for _ in range(n)]
    
    # We want to add edges to minimize the max triangle-free set
    # Strategy: add edges that are part of many potential K4s
    
    # First, add a base structure - a bipartite-like graph between two halves
    # Split vertices into groups
    # Try different partitions
    
    random.seed(42)
    
    best_graph = None
    best_score = float('inf')
    
    for trial in range(50):
        # Random 3-partition
        parts = [list(range(i*6, (i+1)*6)) for i in range(3)]
        
        # Shuffle within parts
        for p in parts:
            random.shuffle(p)
        
        # Create complete multipartite graph
        adj = [[False]*n for _ in range(n)]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = True
                        adj[b][a] = True
        
        # Now add some edges within parts to reduce triangle-free sets
        # But be careful not to create K4s
        for part_idx in range(3):
            part = parts[part_idx]
            # Add edges within the part, checking K4-freeness
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    u, v = part[i], part[j]
                    if not adj[u][v]:
                        # Check if adding (u,v) creates a K4
                        # A K4 would need two other vertices w,x such that
                        # all edges among u,v,w,x exist
                        # Since u,v are in the same part, they have no edges to each other yet
                        # Adding (u,v) means we need to check if there exist w,x in other parts
                        # such that u-w, u-x, v-w, v-x, w-x all exist
                        # In a complete multipartite graph, any w in part_j and x in part_k
                        # (j,k != part_idx) will have all these edges
                        # So if there are at least 2 vertices in other parts combined,
                        # and both are connected to u and v (which they are in complete multipartite),
                        # then adding (u,v) creates a K4 if there exist w in one other part and x in another
                        # with w-x edge (which exists in complete multipartite)
                        
                        # Actually in complete multipartite, any two vertices in different parts are connected.
                        # So if we add (u,v) within part_idx, and there exist w in part_a and x in part_b
                        # (a,b != part_idx), then u-v, u-w, u-x, v-w, v-x, w-x all exist -> K4.
                        # So we can only add (u,v) if the graph is not complete multipartite with other parts.
                        
                        # For now, skip adding intra-part edges in complete multipartite
                        pass
        
        # Evaluate
        if _is_k4_free(adj):
            score = _max_triangle_free_size(adj)
            if score < best_score:
                best_score = score
                best_graph = adj
    
    if best_graph is None:
        # Fallback: Turán graph
        best_graph = [[False]*n for _ in range(n)]
        parts = [range(0,6), range(6,12), range(12,18)]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        best_graph[a][b] = True
                        best_graph[b][a] = True
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i][j]:
                edges.append([i, j])
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a better graph
    result = _construct_greedy_k4_free()
    
    # Verify K4-free
    n = 18
    adj = [[False]*n for _ in range(n)]
    for u, v in result:
        if u != v:
            adj[u][v] = True
            adj[v][u] = True
    
    if not _is_k4_free(adj):
        # Fallback to Turán
        result = [[a, b] for i in range(3) for j in range(i+1, 3)
                  for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen, but just in case
        result = [[a, b] for i in range(3) for j in range(i+1, 3)
                  for a in range(i*6, (i+1)*6) for b in range(j*6, (j+1)*6)]
    
    return result

# EVOLVE-BLOCK-END
