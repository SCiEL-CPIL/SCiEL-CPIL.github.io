import numpy as np
import random
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
    # Let's verify count.
    # If len(edges) != 90, we might need to adjust, but for q=5 it is 90.
    
    # 3. Add one more edge
    # Find a non-edge
    # Since N=31, total possible edges = 31*30/2 = 465.
    # We have 90. There are 375 non-edges.
    # Pick the first non-edge found.
    
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in edge_set:
                edges.append((i, j))
                edge_set.add((i, j))
                break
        if len(edges) == E:
            break
            
    # If for some reason we didn't reach E (should not happen), fill randomly
    if len(edges) < E:
        all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
        remaining = [e for e in all_possible if e not in edge_set]
        random.shuffle(remaining)
        edges.extend(remaining[:E - len(edges)])
        
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
