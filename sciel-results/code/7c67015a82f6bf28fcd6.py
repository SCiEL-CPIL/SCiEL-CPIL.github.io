import random
import time
from itertools import combinations

def has_c5(n, edges):
    """Check if the graph contains a cycle of length 5."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Simpler check for small n (<=35):
    # Iterate over all combinations of 5 vertices.
    # Check if they form a C5.
    # 35C5 = 324632. This is fast enough.
    
    for verts in combinations(range(n), 5):
        v0, v1, v2, v3, v4 = verts
        # Build induced subgraph edges
        sub_edges = set()
        for i in range(5):
            for j in range(i+1, 5):
                if verts[j] in adj[verts[i]]:
                    sub_edges.add((verts[i], verts[j]))
        
        if len(sub_edges) == 5:
            # Check if it's a cycle
            # Degrees must all be 2
            degs = [0]*5
            for u, v in sub_edges:
                idx_u = verts.index(u)
                idx_v = verts.index(v)
                degs[idx_u] += 1
                degs[idx_v] += 1
            if all(d == 2 for d in degs):
                return True
    return False

def max_independent_set_size(n, edges):
    """Find the size of the maximum independent set."""
    # Greedy construction of independent set
    best = 0
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for _ in range(10):
        remaining = list(range(n))
        current = []
        while remaining:
            idx = random.randint(0, len(remaining)-1)
            v = remaining.pop(idx)
            current.append(v)
            remaining = [u for u in remaining if u not in adj[v]]
        if len(current) > best:
            best = len(current)
            if best >= 9:
                return best
    return best

def construct():
    # Construct G as the complement of a bipartite graph H.
    # Let H be the complete bipartite graph K_{18, 17} on 35 vertices.
    # H is bipartite, so it is C5-free.
    # The clique number of H is 2, which is <= 8.
    # Therefore, G = complement(H) has no C5 (since H is C5-free? No, C5 is not self-complementary in that sense)
    # Wait, the condition is G must be C5-free.
    # If H is bipartite, G is the union of two cliques.
    # A union of cliques has no cycles, so it is C5-free.
    # The independence number of G is the clique number of H, which is 2.
    # 2 <= 8.
    # So G is valid.
    
    n = 35
    edges = []
    
    # G consists of two disjoint cliques:
    # Clique 1: vertices 0 to 17
    # Clique 2: vertices 18 to 34
    
    # Generate edges for Clique 1
    for i in range(18):
        for j in range(i + 1, 18):
            edges.append([i, j])
            
    # Generate edges for Clique 2
    for i in range(18, 35):
        for j in range(i + 1, 35):
            edges.append([i, j])
            
    return {'n': n, 'edges': edges}
