import time
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    """
    Constructs a triangle-free graph on n vertices with no independent set of 10.
    Strategy:
    1. Deterministically construct the Polarity Graph of PG(2, 5) (n=31).
    2. Verify triangle-freeness and independence number.
    3. If valid, return n=31.
    4. If invalid due to triangles, remove edges to break triangles and re-check alpha.
    5. If alpha becomes too large, fall back to Andrasfai graph (n=29) or local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    def time_up():
        return time.time() > deadline

    def build_adj_from_edges(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def is_triangle_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    def has_independent_set(adj, n, target_size=10):
        def dfs(candidates, size):
            if size >= target_size:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target_size:
                return False
            
            v_bit = candidates & (-candidates)
            v = v_bit.bit_length() - 1
            rest = candidates ^ v_bit
            
            if dfs(rest & ~adj[v], size + 1):
                return True
            if dfs(rest, size):
                return True
            return False
        
        return dfs((1 << n) - 1, 0)

    def andrasfai_adj(n):
        # Fallback to known valid construction
        # The seed used (j-i) % n % 3 == 1 for n=29
        # This specific construction is known to be triangle-free for n=3k+2?
        # Let's stick to the seed's proven n=29 case.
        if n != 29:
            return None
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # 1. Try Polarity Graph of PG(2, 5)
    # Points of PG(2, 5) are 1D subspaces of F_5^3
    # We can represent them by their canonical representative:
    # (1, y, z), (0, 1, z), (0, 0, 1)
    q = 5
    points = []
    # Type 1: x=1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    # Type 2: x=0, y=1
    for z in range(q):
        points.append((0, 1, z))
    # Type 3: x=0, y=0, z=1
    points.append((0, 0, 1))
    
    n_polarity = len(points) # 25 + 5 + 1 = 31
    
    # Define a bilinear form B(u, v)
    # We want a symmetric form.
    # B(u, v) = u[0]*v[1] + u[1]*v[0] + u[2]*v[2]
    # This form is non-degenerate?
    # Matrix:
    # 0 1 0
    # 1 0 0
    # 0 0 1
    # Det = -1 = 4 != 0 in F_5. So non-degenerate.
    
    def B(u, v):
        return (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % q

    adj_poly = [0] * n_polarity
    for i in range(n_polarity):
        for j in range(i + 1, n_polarity):
            if B(points[i], points[j]) == 0:
                adj_poly[i] |= (1 << j)
                adj_poly[j] |= (1 << i)
    
    # Check if valid
    valid_poly = False
    if is_triangle_free(adj_poly, n_polarity) and not has_independent_set(adj_poly, n_polarity):
        valid_poly = True
        best_n = n_polarity
        best_adj = adj_poly
        best_edges = to_edges(adj_poly, n_polarity)
    else:
        # If not valid, try to repair by removing edges that form triangles
        # This increases alpha, so check carefully
        if not time_up():
            adj_repaired = adj_poly[:]
            # Find all triangles and remove one edge from each
            # Simple heuristic: iterate edges, if it forms a triangle, remove it
            # This might be too aggressive. 
            # Let's just fall back to Andrasfai for now if Polarity fails immediately.
            # But let's try a quick repair:
            for u in range(n_polarity):
                for v in range(u + 1, n_polarity):
                    if adj_repaired[u] & (1 << v):
                        if adj_repaired[u] & adj_repaired[v]:
                            # Remove edge (u, v)
                            adj_repaired[u] &= ~(1 << v)
                            adj_repaired[v] &= ~(1 << u)
            
            if is_triangle_free(adj_repaired, n_polarity) and not has_independent_set(adj_repaired, n_polarity):
                valid_poly = True
                best_n = n_polarity
                best_adj = adj_repaired
                best_edges = to_edges(adj_repaired, n_polarity)
        
        if not valid_poly:
            # Fallback to Andrasfai n=29
            adj29 = andrasfai_adj(29)
            if adj29 and is_triangle_free(adj29, 29) and not has_independent_set(adj29, 29):
                best_n = 29
                best_adj = adj29
                best_edges = to_edges(adj29, 29)
            else:
                best_n = 1
                best_adj = [0] * 1
                best_edges = []

    # If we have a valid graph with n < 31, we might try to extend it?
    # For now, return the best found.
    # The Polarity graph is a strong deterministic candidate for n=31.
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Best n: {result['n']}")
