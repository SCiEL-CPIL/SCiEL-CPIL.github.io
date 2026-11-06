import time
import random

def construct():
    """
    Constructs a K2,3-free graph on 20 vertices maximizing edges.
    Strategy:
    1. Start with the Clebsch graph (16 vertices, 48 edges, K2,3-free) + 4 isolated vertices.
    2. Use a greedy algorithm to add any remaining edges that do not violate the K2,3 constraint.
    3. The constraint check is optimized using the fact that adding (u,v) only affects 
       common[u][w] for w in N(v) and common[v][w] for w in N(u).
    """
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    # --- Step 1: Generate Clebsch Graph on 16 vertices ---
    # Clebsch graph is the folded 5-cube.
    # Vertices 0..15.
    # Map Q5 vertices (0..31) to Clebsch vertices (0..15).
    # Representative is min(x, x ^ 31) ? No, antipodal in Q5 is x ^ 31.
    # Let's just use the first 16 and map the rest.
    # Actually, simpler: 
    # Vertices of Clebsch are pairs {x, x^11111} from Q5.
    # We can label them by the min of the pair.
    
    def get_clebsch_rep(x):
        return min(x, x ^ 31)
    
    clebsch_edges = set()
    # Generate edges of Q5
    for u in range(32):
        for b in range(5):
            v = u ^ (1 << b)
            if u < v:
                cu = get_clebsch_rep(u)
                cv = get_clebsch_rep(v)
                if cu != cv:
                    clebsch_edges.add((cu, cv))
                else:
                    # This happens if u and v are antipodal? No, distance 1.
                    # In Q5, distance is 1. Antipodal distance is 5.
                    # So cu != cv always for edges in Q5.
                    pass
    
    # Add 4 isolated vertices 16, 17, 18, 19
    # Initialize adjacency and common neighbor counts
    adj = [[False] * n for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    
    # Add Clebsch edges
    for u, v in clebsch_edges:
        adj[u][v] = True
        adj[v][u] = True
        # Update common neighbors
        # For edge (u, v), u becomes a common neighbor of v and all neighbors of v (excluding u)
        # And v becomes a common neighbor of u and all neighbors of u (excluding v)
        # But we need to do this incrementally or recompute.
        # Since we are building from scratch, let's just add edges and recompute common at the end of init?
        # Or update carefully.
        
    # Recompute common neighbors for the initial graph
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            common[u][v] = count
            common[v][u] = count
            
    # --- Step 2: Greedy Edge Addition ---
    # List of potential edges to try
    # We can try to add any non-edge
    # To be efficient, we can shuffle or just iterate.
    # Since n=20, there are 190 edges.
    
    # We will iterate until no more edges can be added
    # Or until deadline
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                
                # Check if adding (u, v) is valid
                # Condition:
                # 1. For all w in N(v) (excluding u, but u not in N(v) yet), common[u][w] < 2
                # 2. For all w in N(u) (excluding v, but v not in N(u) yet), common[v][w] < 2
                
                valid = True
                # Check neighbors of v
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w]:
                        if common[u][w] >= 2:
                            valid = False
                            break
                if not valid:
                    continue
                    
                # Check neighbors of u
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w]:
                        if common[v][w] >= 2:
                            valid = False
                            break
                            
                if valid:
                    # Add edge (u, v)
                    adj[u][v] = True
                    adj[v][u] = True
                    changed = True
                    
                    # Update common neighbor counts
                    # u becomes a common neighbor of v and all w in N(v)
                    # v becomes a common neighbor of u and all w in N(u)
                    
                    # For each w in N(v):
                    # common[u][w] increases by 1 (because u is now neighbor of v, and u was already neighbor of w? No.
                    # u is neighbor of w? 
                    # The new common neighbor for pair (u, w) is v?
                    # v is neighbor of u (new).
                    # v is neighbor of w (yes, w in N(v)).
                    # So v is a new common neighbor of u and w.
                    # So common[u][w] += 1.
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[v][w]:
                            common[u][w] += 1
                            common[w][u] += 1
                            
                    # For each w in N(u):
                    # u is a new common neighbor of v and w.
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[u][w]:
                            common[v][w] += 1
                            common[w][v] += 1

    # --- Step 3: Convert to list of edges ---
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
                
    return edges

if __name__ == "__main__":
    # Simple test
    edges = construct()
    print(f"Edges: {len(edges)}")
