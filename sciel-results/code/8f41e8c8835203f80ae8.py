import numpy as np
import time

N = 31
E = 91
DEADLINE_SECONDS = 115.0

def construct():
    """
    Constructs a graph with 31 vertices and 91 edges minimizing the number of C4s.
    Uses the polarity graph of the projective plane PG(2,5) which has 90 edges and 0 C4s.
    Adds one non-edge to reach 91 edges. Since the base graph is C4-free,
    any non-edge has at most 1 common neighbor, so adding it creates 0 C4s.
    """
    
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS

    # 1. Generate points of PG(2,5)
    # Points are 1-dim subspaces of F_5^3
    # We represent them by their canonical form: first non-zero coordinate is 1.
    # Or simply enumerate representatives.
    
    points = []
    # Case 1: z != 0. Scale so z=1.
    # (x, y, 1) for x, y in F_5
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
            
    # Case 2: z == 0, y != 0. Scale so y=1.
    # (x, 1, 0) for x in F_5
    for x in range(5):
        points.append((x, 1, 0))
        
    # Case 3: z == 0, y == 0, x != 0. Scale so x=1.
    # (1, 0, 0)
    points.append((1, 0, 0))
    
    # Total 25 + 5 + 1 = 31 points.
    # Map point index to vector
    # points[i] is the vector for vertex i.
    
    edges = []
    edge_set = set()
    
    # 2. Construct polarity graph
    # Edge (i, j) if dot(points[i], points[j]) == 0 mod 5
    # This specific dot product defines a polarity.
    for i in range(N):
        for j in range(i + 1, N):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                edges.append((i, j))
                edge_set.add((i, j))
                
    # We should have 90 edges.
    # If not, something is wrong with the construction, but theory says 90.
    
    # 3. Add one more edge to minimize new C4s
    # The base graph is C4-free (0 C4s).
    # Adding an edge (u, v) creates C4s equal to the number of common neighbors of u and v.
    # In a polarity graph of PG(2,q), any two non-adjacent points have exactly 1 common neighbor.
    # Therefore, any non-edge addition creates exactly 1 C4.
    # So the minimum possible C4 count is 1.
    
    # Let's verify this assumption and find the edge with minimum common neighbors.
    # If the minimum is > 1, we pick the best one. If it is 1, we pick the first one.
    # Since the theoretical lower bound for adding 1 edge to a C4-free graph is 1 (if min common neighbors is 1),
    # and we know polarity graphs have this property, the result should be 1.
    
    # Build adjacency list for speed
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_edge = None
    min_common = float('inf')
    
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edge_set:
                continue
            common = len(adj[i] & adj[j])
            if common < min_common:
                min_common = common
                best_edge = (i, j)
                if min_common == 0:
                    break
        if min_common == 0:
            break
            
    if best_edge:
        edges.append(best_edge)
        
    # Fallback if something went wrong (should not happen)
    if len(edges) < E:
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in edge_set:
                    edges.append((i, j))
                    edge_set.add((i, j))
                    break
            if len(edges) == E:
                break
        
    return edges

# Helper to count C4s for verification (not used in final return but good for sanity check if run locally)
def count_c4(adj):
    A2 = adj @ adj
    i, j = np.triu_indices(N, k=1)
    common_counts = A2[i, j]
    terms = common_counts * (common_counts - 1) // 2
    total = np.sum(terms)
    return total // 2

if __name__ == "__main__":
    # Local test
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((N, N), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    c4s = count_c4(adj)
    print(f"C4s: {c4s}")
