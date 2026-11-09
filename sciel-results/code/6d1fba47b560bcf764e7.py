import numpy as np

def construct():
    n = 31
    # Generate vertices of PG(2,5)
    # 25 affine points: (x, y, 1)
    # 5 points at infinity: (x, 1, 0)
    # 1 point at infinity: (1, 0, 0)
    
    vertices = []
    # Affine
    for x in range(5):
        for y in range(5):
            vertices.append((x, y, 1))
    # Infinity y
    for x in range(5):
        vertices.append((x, 1, 0))
    # Infinity x
    vertices.append((1, 0, 0))
    
    # Function to check adjacency
    def are_adjacent(v1, v2):
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        return (x1 * x2 + y1 * y2 - z1 * z2) % 5 == 0

    # Build adjacency list and edge list
    edges = []
    adj = {i: set() for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            if are_adjacent(vertices[i], vertices[j]):
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    # Check edge count
    # Should be 90
    if len(edges) != 90:
        # Fallback if something is wrong, though it should be 90
        # Just return first 91 lexicographic for safety? No, let's trust the math.
        # But for robustness in this environment, if it fails, we might need a fallback.
        # However, the math is solid.
        pass

    # Find the non-edge that minimizes C4 count
    min_count = float('inf')
    best_edge = None
    
    # Precompute common neighbors or just iterate
    # For each non-edge (u, v), count paths u-w-x-v
    # w in N(u), x in N(v), w ~ x, w != x
    
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                continue # Already an edge
            
            count = 0
            for w in adj[i]:
                for x in adj[j]:
                    if w == x:
                        continue
                    if x in adj[w]: # w ~ x
                        count += 1
                        
            if count < min_count:
                min_count = count
                best_edge = [i, j]
                
    if best_edge is not None:
        edges.append(best_edge)
        
    return edges
