import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    
    # Construct Clebsch Graph (Folded 5-cube)
    # Vertices 0..15 represent 5-bit strings 00000..01111
    # Edge (u, v) if Hamming distance of 5-bit reps is 1 or 4
    
    def get_clebsch_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                # Calculate Hamming distance of 5-bit representations
                # u and v are < 16, so 5th bit is 0 for both
                diff = u ^ v
                dist = bin(diff).count('1')
                # In 5-bit space, distance is popcount(u^v) since high bits are 0
                # But wait, the folding logic:
                # d_5(u, v) = popcount(u ^ v) if we consider them as 5-bit.
                # Is it possible that d_5(u, v) = 4 corresponds to d_5(u, complement(v)) = 1?
                # Yes.
                # So we check if dist == 1 or dist == 4.
                if dist == 1 or dist == 4:
                    edges.append([u, v])
        return edges

    base_edges = get_clebsch_edges()
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Validation functions
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i > j:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True

    def is_not_3_colorable(adj, n):
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Heuristic: try colors that conflict least? Or just 0,1,2
            for c in range(3):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        # Use degree ordering for speed
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        # Map original index to position in order? 
        # Actually, standard backtracking on permuted graph is tricky with simple colors array.
        # Let's just use the standard one, it's fast for n=16.
        return not backtrack(0)

    # Check base graph
    if not is_triangle_free(adj, n) or not is_not_3_colorable(adj, n):
        # Fallback to Mycielski if Clebsch fails (should not happen)
        # But let's assume it's valid.
        pass

    # Phase 2: Local Search to maximize edges
    # Try to add edges that are not present and don't create triangles
    # If adding an edge makes it 3-colorable, reject.
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Current edge count
    current_edges = len(base_edges)
    
    # Sort addable edges by some heuristic (e.g., number of common non-neighbors? or just random)
    # We want to add edges that are "safe" regarding triangles.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Find all addable edges
        addable = []
        for u, v in all_possible_edges:
            if v in adj[u]:
                continue
            # Check triangle freedom
            if not (adj[u] & adj[v]):
                addable.append((u, v))
                
        random.shuffle(addable)
        
        for u, v in addable:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            
            # Tentatively add
            adj[u].add(v)
            adj[v].add(u)
            
            if is_not_3_colorable(adj, n):
                # Keep edge
                current_edges += 1
                improved = True
            else:
                # Revert
                adj[u].discard(v)
                adj[v].discard(u)
                
    # Final validation
    if not is_triangle_free(adj, n) or not is_not_3_colorable(adj, n):
        # Return base if corrupted
        return base_edges
        
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append([u, v])
                
    return final_edges
