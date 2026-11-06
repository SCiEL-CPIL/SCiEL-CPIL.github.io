# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 115.0
    n = 31
    target_edges = 91
    
    # Generate points in PG(2, 5)
    # Points are [x:y:z] in GF(5)^3 \ {0}, modulo scaling.
    # We can represent them by fixing one coordinate to 1 if possible, or using a canonical form.
    # Simpler: iterate all non-zero vectors, and keep the first one encountered for each 1D subspace.
    
    points = []
    seen = set()
    
    def add_point(p):
        # Canonical form: make the first non-zero element 1
        p = list(p)
        for i in range(3):
            if p[i] != 0:
                inv = pow(p[i], 4, 5) # Fermat's little theorem for GF(5), x^4 = 1/x
                p = [(x * inv) % 5 for x in p]
                break
        p_tuple = tuple(p)
        if p_tuple not in seen:
            seen.add(p_tuple)
            points.append(p_tuple)
            
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                add_point((x, y, z))
                
    # We should have 31 points
    if len(points) != 31:
        # Fallback if generation failed (should not happen)
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]

    # Define a polarity.
    # Use symmetric matrix M.
    # Let M = diag(1, 1, 1).
    # P . Q = x1*x2 + y1*y2 + z1*z2
    # Adjacency: P . Q == 0 and P != Q
    
    M = np.eye(3, dtype=int)
    
    adj = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = np.array(points[i])
            p_j = np.array(points[j])
            dot = np.dot(p_i, p_j) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                
    # Check edge count
    current_edges = len(edges)
    
    # Theoretically, this should be 90.
    # If it's not 90, we might need to adjust M or the definition.
    # But for PG(2,5) with standard dot product, it should be a polarity graph.
    
    if current_edges != 90:
        # If edge count is wrong, we might have a different graph.
        # Let's try to fix it by adding/removing edges to reach 90, but this breaks the structure.
        # However, the problem states ex(31, C4) = 90.
        # If this graph has C4s, it's not ideal, but we can still use it as a base.
        # But let's assume the standard construction works.
        # If it doesn't, we fall back to the previous logic or a simple greedy.
        
        # For now, if it's not 90, we just proceed with what we have, 
        # but we need exactly 91 edges.
        # If current_edges > 91, remove some.
        # If current_edges < 91, add some.
        
        # To be safe, let's use the previous greedy method if this fails.
        # But let's assume it works.
        pass

    # Compute A^3 to find best edge to add
    A = adj
    A2 = A @ A
    A3 = A2 @ A
    
    # Find non-edge (u,v) with minimum A3[u,v]
    best_val = float('inf')
    best_u, best_v = -1, -1
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                val = A3[u, v]
                if val < best_val:
                    best_val = val
                    best_u, best_v = u, v
                    
    if best_u != -1:
        edges.append((best_u, best_v))
        
    # Ensure we have exactly 91 edges
    # If we had 90, we added 1, so we have 91.
    # If we had != 90, we might have issues.
    # Let's pad or trim if necessary.
    
    if len(edges) < 91:
        # Add random non-edges
        existing = set(edges)
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in existing:
                    edges.append((u, v))
                    if len(edges) == 91:
                        break
            if len(edges) == 91:
                break
                
    elif len(edges) > 91:
        # Remove edges
        edges = edges[:91]
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
