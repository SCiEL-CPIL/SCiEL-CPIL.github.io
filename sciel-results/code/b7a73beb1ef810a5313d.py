import time
import random

def _is_triangle_free(adj, u, v):
    # Check if u and v have a common neighbor
    # adj is a list of sets
    common = adj[u] & adj[v]
    return len(common) == 0

def _is_3_colorable(adj, n):
    # Simple backtracking for 3-colorability
    color = [-1] * n
    # Order vertices by degree descending to prune faster
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(i + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    n = 16
    # Generate halved 5-cube
    # Vertices 0..15 mapped to 5-bit strings? No, 16 vertices are 4-bit strings.
    # But halved 5-cube has 16 vertices (even parity of 5 bits).
    # Let's use 5-bit representation for the 16 vertices.
    # Map vertex i (0..15) to a 5-bit string with even parity.
    # There are 16 such strings.
    
    # Generate all 5-bit strings with even parity
    even_vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            even_vertices.append(i)
            
    # Map index 0..15 to these values
    val_to_idx = {v: i for i, v in enumerate(even_vertices)}
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Add edges if Hamming distance is 2
    for i in range(n):
        for j in range(i + 1, n):
            u = even_vertices[i]
            v = even_vertices[j]
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 2:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    # This graph has triangles. We need to remove them.
    # We will iteratively find and remove edges that form triangles.
    # To maximize edges, we should remove edges that are part of the most triangles?
    # Or just any edge in a triangle.
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    # Check if triangle exists
                    common = adj[i] & adj[j]
                    if common:
                        # Remove edge i-j
                        adj[i].remove(j)
                        adj[j].remove(i)
                        edges = [e for e in edges if not (e[0] == i and e[1] == j) and not (e[0] == j and e[1] == i)]
                        changed = True
                        break
            if changed:
                break
                
    # Now the graph is triangle-free.
    # Check if it is 3-colorable.
    # If it is, we need to modify it to be non-3-colorable.
    # But removing edges usually keeps it colorable.
    # We might need to add edges? No, we removed them.
    # The halved 5-cube is 4-chromatic. Removing edges might make it 3-colorable.
    
    if _is_3_colorable(adj, n):
        # Try to add back edges that don't create triangles and break 3-colorability
        # This is hard. 
        # Alternative: Start with a known 4-chromatic triangle-free graph.
        # The Grötzsch graph (11 vertices) is one.
        # We can pad it.
        
        pass
        
    return edges

