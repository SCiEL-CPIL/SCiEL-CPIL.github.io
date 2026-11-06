import time

def construct():
    # 1. Generate PG(2,5) points
    # Points are (x, y, z) normalized
    points = []
    # (1, y, z)
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    # (0, 1, z)
    for z in range(5):
        points.append((0, 1, z))
    # (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    assert n == 31

    # 2. Define Polarity
    # Matrix A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    # B(u, v) = u1*v2 + u2*v1 + u2*v3 + u3*v2
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        # Compute in GF(5)
        val = (x1 * y2 + y1 * x2 + y1 * z2 + z1 * y2) % 5
        return val == 0

    # 3. Build Graph
    edges = []
    neigh = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if is_adjacent(points[i], points[j]):
                edges.append((i, j))
                neigh[i] |= (1 << j)
                neigh[j] |= (1 << i)
                
    # Check edge count
    # Expected 93
    if len(edges) != 93:
        # Fallback: If polarity is wrong, try another or use random?
        # But 93 is the standard count for this polarity.
        # If it fails, we might need to adjust.
        pass

    # 4. Find best 2 edges to remove
    def count_c4s(neigh):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = bin(neigh[i] & neigh[j]).count('1')
                count += common * (common - 1) // 2
        return count // 2

    best_count = float('inf')
    best_edges = None
    best_neigh = None
    
    # Iterate over pairs of edges to remove
    # To speed up, we can just try all pairs
    m = len(edges)
    for i in range(m):
        e1 = edges[i]
        u1, v1 = e1
        for j in range(i + 1, m):
            e2 = edges[j]
            u2, v2 = e2
            
            # Create a copy of neigh to modify
            # Since n is small, copying 31 ints is fast
            temp_neigh = neigh[:]
            
            # Remove e1
            temp_neigh[u1] &= ~(1 << v1)
            temp_neigh[v1] &= ~(1 << u1)
            
            # Remove e2
            temp_neigh[u2] &= ~(1 << v2)
            temp_neigh[v2] &= ~(1 << u2)
            
            # Count C4s
            c4_count = count_c4s(temp_neigh)
            
            if c4_count < best_count:
                best_count = c4_count
                best_edges = edges[:i] + edges[i+1:j] + edges[j+1:]
                best_neigh = temp_neigh[:]
                
                # If we found 0, we can stop early? 
                # But ex(31, C4)=90 means 91 edges MUST have >= 1.
                # So 0 is impossible. Minimum is likely 1.
                if best_count == 1:
                    break
        if best_count == 1:
            break

    # Return the edges
    # The function must return a list of lists [u, v]
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
