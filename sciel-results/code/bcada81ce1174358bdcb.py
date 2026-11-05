import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Parameters for PG(2, q)
    # We try q=7 first. n = 57.
    q = 7
    
    # Generate points of PG(2, q)
    # Points are (x, y, z) not all 0, modulo scaling.
    # We represent them as tuples (x, y, z) with z=1 or z=0 (points at infinity)
    points = []
    # z = 1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # z = 0, x != 0
    for x in range(1, q):
        for y in range(q):
            # Normalize so x=1? 
            # In PG(2,q), (x,y,0) is equivalent to (1, y/x, 0)
            # So we can just list (1, k, 0) for k in F_q
            # But wait, x can be any non-zero.
            # The points at infinity are (1, k, 0) for k in F_q.
            # There are q such points.
            pass
            
    # Correct enumeration:
    # Total q^2 + q + 1
    # Affine part: (x, y, 1) for x,y in F_q. (q^2 points)
    # Line at infinity: (1, k, 0) for k in F_q. (q points)
    # Point (0, 1, 0). (1 point)
    
    pts = []
    for x in range(q):
        for y in range(q):
            pts.append((x, y, 1))
    for k in range(q):
        pts.append((1, k, 0))
    pts.append((0, 1, 0))
    
    n = len(pts)
    if n != q*q + q + 1:
        # Fallback to smaller q if logic error
        pass
        
    # Define Polarity
    # A = diag(1, 1, -1)
    # u ~ v if u0*v0 + u1*v1 - u2*v2 == 0
    
    def is_adj(u, v):
        return (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q == 0

    # Build Graph
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if is_adj(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    # Function to find Independent Set of size k
    # Using simple backtracking/branch and bound
    def find_is_size_k(k, graph, n):
        # graph is list of sets
        # We want to find a set of k vertices with no edges between them
        # Sort vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: -len(graph[x]))
        
        current = []
        
        def backtrack(idx, current_set):
            if len(current_set) == k:
                return current_set[:]
            if idx == n:
                return None
            # Pruning: if not enough vertices left
            if len(current_set) + (n - idx) < k:
                return None
            # Pruning: if we can't add more
            # Check if current vertex can be added
            v = order[idx]
            # Check if v is compatible with all in current_set
            compatible = True
            for u in current_set:
                if v in graph[u]:
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                res = backtrack(idx + 1, current_set)
                if res:
                    return res
                current_set.pop()
            
            # Try skipping v
            return backtrack(idx + 1, current_set)
            
        return backtrack(0, [])

    # Check validity
    # We want alpha <= 10.
    # If find_is_size_k(11) returns None, alpha <= 10.
    # If it returns a set, alpha >= 11.
    
    removed = set()
    
    # Loop to remove vertices if alpha > 10
    while time.time() < deadline:
        # Build subgraph on remaining vertices
        # To simplify, we just check the current graph structure
        # If we remove vertices, we need to update adjacency.
        
        # For efficiency, let's just check if the full graph has IS 11.
        # If yes, remove a vertex from it.
        
        # We need a dynamic graph representation for removal.
        # Let's use a list of active vertices.
        active = [i for i in range(n) if i not in removed]
        
        # Map active indices to original
        # This is getting complex to re-run backtrack on subgraph.
        # Alternative: Just check the full graph initially.
        # If invalid, remove vertices and re-check full graph on remaining?
        # No, IS of subgraph is IS of full graph.
        # So we can just search for IS 11 in the full graph,
        # but restricting to 'active' vertices.
        
        # Modify find_is to take a mask
        def find_is_masked(k, mask):
            # mask is a set of allowed vertices
            order = sorted([v for v in range(n) if v in mask], key=lambda x: -len(adj[x]))
            
            def backtrack(idx, current_set):
                if len(current_set) == k:
                    return current_set[:]
                if idx == len(order):
                    return None
                if len(current_set) + (len(order) - idx) < k:
                    return None
                
                v = order[idx]
                compatible = True
                for u in current_set:
                    if v in adj[u]:
                        compatible = False
                        break
                
                if compatible:
                    current_set.append(v)
                    res = backtrack(idx + 1, current_set)
                    if res:
                        return res
                    current_set.pop()
                
                return backtrack(idx + 1, current_set)
            
            return backtrack(0, [])

        mask = set(active)
        is_found = find_is_masked(11, mask)
        
        if is_found is None:
            # Valid
            break
        else:
            # Found IS of size 11. Remove one vertex.
            # Heuristic: Remove vertex with highest degree in the IS?
            # Or just the first one.
            v_to_remove = is_found[0]
            removed.add(v_to_remove)
            # Update active list
            active = [i for i in active if i not in removed]
            
            if len(active) <= 10:
                # If we remove too many, we might drop below seed?
                # But n=57, removing 10 leaves 47. 47 > 30.
                pass

    # Construct final edge list
    final_edges = []
    for i in range(n):
        if i in removed:
            continue
        for j in adj[i]:
            if i < j and j not in removed:
                final_edges.append([i, j])
                
    # Renumber vertices to 0..m-1
    active_nodes = [i for i in range(n) if i not in removed]
    node_map = {old: new for new, old in enumerate(active_nodes)}
    
    mapped_edges = []
    for u, v in final_edges:
        mapped_edges.append([node_map[u], node_map[v]])
        
    return {'n': len(active_nodes), 'edges': mapped_edges}
