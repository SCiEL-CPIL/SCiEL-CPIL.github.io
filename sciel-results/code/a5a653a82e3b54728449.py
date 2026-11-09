"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Attempt to construct the Polarity Graph of PG(2, 5)
    # This graph has n=31 vertices, is C4-free, and has alpha=10.
    try:
        n, edges = _build_pg2_5_polarity()
        # Verify properties
        if _is_c4_free(n, edges) and _max_independent_set_size(n, edges) < 11:
            return {'n': n, 'edges': edges}
    except Exception:
        pass
    
    # Fallback: 10 disjoint triangles (n=30)
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def _build_pg2_5_polarity():
    q = 5
    # Generate points of PG(2, q)
    # Points are (x, y, z) with not all 0, normalized so first non-zero is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    points.append((1, (y * pow(x, q-2, q)) % q, (z * pow(x, q-2, q)) % q))
                elif y != 0:
                    points.append((0, 1, (z * pow(y, q-2, q)) % q))
                else:
                    points.append((0, 0, 1))
    
    n = len(points)
    if n != 31:
        raise ValueError(f"Expected 31 points, got {n}")
    
    # Define polarity: x . y = 0
    # Using the standard dot product in GF(5)
    # Note: This creates a self-polar graph.
    # Edge if dot product is 0.
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            dot = (x1 * x2 + y1 * y2 + z1 * z2) % q
            if dot == 0:
                edges.append([i, j])
                
    return n, edges

def _is_c4_free(n, edges):
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check for C4: u-v-w-x-u
    # For each pair of vertices (u, v), if they are adjacent,
    # they must not have a common neighbor? No, that's for triangles.
    # C4 condition: No two vertices have more than 1 common neighbor.
    # If u and v have two common neighbors w and x, then u-w-v-x-u is a C4.
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u].intersection(adj[v])
            if len(common) > 1:
                return False
    return True

def _max_independent_set_size(n, edges):
    # Branch and bound for maximum independent set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_size = 0
    
    def dfs(remaining, current_size):
        nonlocal max_size
        if not remaining:
            return
        if current_size + len(remaining) <= max_size:
            return
        # Heuristic: pick vertex with highest degree in remaining?
        # Or just first vertex
        v = next(iter(remaining))
        # Option 1: Include v
        new_remaining = remaining - {v} - adj[v]
        dfs(new_remaining, current_size + 1)
        # Option 2: Exclude v
        new_remaining = remaining - {v}
        dfs(new_remaining, current_size)
        
    dfs(set(range(n)), 0)
    return max_size

# EVOLVE-BLOCK-END
