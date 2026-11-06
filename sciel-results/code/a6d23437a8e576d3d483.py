import random
import itertools

def construct():
    random.seed(42)
    n = 16
    
    # Start with a sparse base and greedily add edges
    # Avoid K_{4,4} to guarantee Q3-free
    
    adj = {i: set() for i in range(n)}
    
    def would_create_k44(u, v):
        """Check if adding edge (u,v) creates a K_{4,4}."""
        # Case 1: u,v in same part of K_{4,4}
        # Need 2 more vertices x,y and a 4-set W⊆V\{u,v,x,y}
        # such that {u,v,x,y} × W is fully connected
        # For each pair (x,y) from remaining vertices:
        rest = [w for w in range(n) if w not in (u, v)]
        for x, y in itertools.combinations(rest, 2):
            # Check if u,v,x,y all connect to each other (within the 4-set)
            # and find 4-set W where all 16 cross-edges exist
            # First check internal edges of {u,v,x,y} - not needed for K_{4,4}
            # K_{4,4} only requires bipartite edges
            # Find common neighbors of u,v,x,y (excluding themselves)
            common = adj[u] & adj[v] & adj[x] & adj[y]
            common.discard(u); common.discard(v); common.discard(x); common.discard(y)
            if len(common) >= 4:
                # Pick any 4 from common
                return True
        
        # Case 2: u,v in different parts
        # Need X (3 verts) and Y (3 verts) disjoint, u∉Y, v∉X
        # ({u}∪X) × ({v}∪Y) fully connected
        # This means:
        #   - X ⊆ N(u)∩N(v), X disjoint from {v}
        #   - Y ⊆ N(u)∩N(v), Y disjoint from {u}
        #   - X and Y disjoint
        #   - All edges: X to v, X to Y, u to Y, u to X (already have u-v)
        #   - Specifically: each x∈X connected to v and all y∈Y
        #     each y∈Y connected to u and all x∈X
        
        common_uv = adj[u] & adj[v]
        common_uv.discard(u); common_uv.discard(v)
        candidates = list(common_uv)
        
        for X in itertools.combinations(candidates, 3):
            Xset = set(X)
            # Y must be from candidates \ X, and each y in Y must connect to all x in X
            # and u must connect to all y in Y (already true since Y⊆N(u))
            # and v must connect to all x in X (already true since X⊆N(v))
            # Need: X × Y fully connected
            valid_Y = [w for w in candidates if w not in Xset and all(w in adj[x] for x in X)]
            if len(valid_Y) >= 3:
                return True
        
        return False
    
    # Greedy: try to add each edge in random order
    all_pairs = list(itertools.combinations(range(n), 2))
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if v not in adj[u]:
            if not would_create_k44(u, v):
                adj[u].add(v)
                adj[v].add(u)
    
    result = [[u, v] for u in range(n) for v in adj[u] if u < v]
    return result
