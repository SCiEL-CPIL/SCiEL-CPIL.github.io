import time
import random
import numpy as np
from itertools import combinations

def _has_c4(adj, n):
    """Check if graph has a C4. adj is a set of frozensets or list of sets."""
    # For each pair of vertices, check if they share 2+ common neighbors
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False

def _min_degree(adj, n):
    return min(len(adj[i]) for i in range(n))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_n = 39
    best_edges = []
    
    # Try to build a C4-free graph with high min degree
    # Strategy: Start with a high-degree graph and remove edges to break C4s
    # while maintaining high minimum degree
    
    # Target: n=46, need min_degree >= 7 (46-39)
    # That's a very sparse requirement! We just need each vertex to have 7+ neighbors.
    
    # Actually, let's think about this differently.
    # We want a graph with NO C4 and min_degree >= n-39.
    # For n=46: min_degree >= 7
    # For n=47: min_degree >= 8
    
    # A random graph G(n, p) has expected degree np.
    # For n=46, p=0.2 gives expected degree ~9.2
    # C4 count in G(n,p) ~ C(n,4)*p^4 ~ (46*45*44*43/24)*p^4
    
    # Let's try a greedy construction:
    # 1. Start with empty graph on n vertices
    # 2. Add edges that don't create a C4, prioritizing vertices with low degree
    # 3. Try to reach min_degree >= n-39
    
    def try_construct(n, target_deg, seed):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(all_edges)
        
        def would_create_c4(u, v):
            """Check if adding edge (u,v) creates a C4."""
            # A C4 containing (u,v) would be u-a-b-v-u or u-a-v-b-u etc.
            # Actually, adding edge (u,v) creates a C4 if there exist a,b 
            # such that u~a, a~b, b~v (and a != v, b != u, a != b)
            # Or more simply: if u and v have 2+ common neighbors, 
            # then adding edge (u,v) doesn't create C4 by itself,
            # but we need to check if the new edge completes a C4.
            
            # Adding edge (u,v) creates a C4 if there exist vertices a, b
            # such that: u~a, a~b, b~v, and a!=v, b!=u, a!=b
            # This means u and v are at distance 2 via a path of length 2
            # Wait, that's not right. A C4 is a cycle of length 4.
            # Adding edge (u,v) creates a C4 if there is a path u-a-b-v of length 3
            # where a and b are distinct and not u or v.
            
            # So we need: exists a in N(u), b in N(v), a != b, a != v, b != u, and a~b
            
            # More efficient: for each common "near-neighbor" pattern
            # Actually, let's just check: for each a in N(u), for each b in N(v),
            # if a != b and a != v and b != u and a in N(b), then C4 exists.
            
            # But this is O(deg(u)*deg(v)) which could be slow.
            # Alternative: adding (u,v) creates C4 iff there exist a,b with
            # u~a, a~b, b~v. So we need a in N(u) \ {v} and b in N(v) \ {u}
            # with a~b and a != b.
            
            nu = adj[u] - {v}
            nv = adj[v] - {u}
            
            # For each a in nu, check if any neighbor of a is in nv
            for a in nu:
                if a in nv:
                    # a is adjacent to both u and v, so u-a-v is a path of length 2
                    # Adding u-v creates triangle, not C4
                    continue
                # Check if any neighbor of a is in nv
                for b in adj[a]:
                    if b in nv and b != a:
                        return True
            
            return False
        
        # Greedy: add edges in random order if they don't create C4
        # But we want high minimum degree, so let's be smarter
        
        # Phase 1: Try to get all vertices to target degree
        # Use a priority queue approach: always add edge to vertex with lowest degree
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            deg_u = len(adj[u])
            deg_v = len(adj[v])
            
            # Only add if it helps the lower-degree vertex
            min_deg = min(deg_u, deg_v)
            if min_deg >= target_deg:
                continue
            
            if not would_create_c4(u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        # Check if we achieved the target
        md = min(len(adj[i]) for i in range(n))
        if md >= target_deg:
            # Verify no C4
            if not _has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    # Try different n values from high to low
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        target_deg = n - 39
        if target_deg < 1:
            target_deg = 1
        
        # Try multiple random seeds
        for seed in range(200):
            if time.time() > deadline:
                break
            
            edges = try_construct(n, target_deg, seed)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                # Found one for this n, try higher n
                break
        
        if time.time() > deadline:
            break
    
    # If we didn't find anything better, return the trivial solution
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
