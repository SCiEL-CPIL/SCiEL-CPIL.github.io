import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    N = 256
    # 1. Identify bad vertices (2v is a perfect square)
    bad = set()
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r == s:
            bad.add(v)
            
    # Valid vertices
    valid_vertices = [v for v in range(1, N + 1) if v not in bad]
    # Map original values to indices 0..M-1
    M = len(valid_vertices)
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build adjacency for "compatible" graph H
    # Edge (i, j) in H if valid_vertices[i] + valid_vertices[j] is NOT a square
    # We want Max Independent Set in Conflict Graph, which is Max Clique in H.
    adj = [0] * M
    
    # Precompute squares up to 2*256 = 512
    squares = set()
    r = 1
    while r * r <= 512:
        squares.add(r * r)
        r += 1
        
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            s = vi + vj
            if s not in squares:
                # Compatible
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Max Clique Search
    best_clique_size = 0
    best_clique_set = 0
    
    # Sort vertices by degree in H (descending) for better pruning
    # Static degree
    degrees = [adj[i].bit_count() for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    
    # We will use a recursive function.
    # To avoid deep recursion limits and overhead, we implement a standard B&B.
    # State: 
    #   P: bitset of candidates
    #   C: current clique (bitset or list)
    
    # Greedy Coloring function for bound
    # Returns (max_color, color_of_vertex)
    # We need to color the subgraph H induced by P.
    # Vertices in P are ordered by 'order' index? 
    # Standard Tomita: Order vertices in P by the color assigned in a greedy coloring of the full graph?
    # Or just sort by degree in P?
    # Let's use a simple greedy coloring on the fly for the bound.
    
    def get_color_bound(P, current_clique_size):
        # If P is empty, bound is 0
        if P == 0:
            return 0
        
        # We need to iterate through vertices in P.
        # To make it fast, we iterate in a fixed order (e.g., index order or degree order).
        # The bound is tight if we pick vertices that are hard to color first.
        # Let's iterate through vertices in P sorted by degree in P (approximated by static degree or popcount(P & adj[v])).
        
        # For simplicity and speed, let's just use the static order 'order' filtered by P.
        # This might be slower to compute but easier.
        # Actually, calculating popcount(P & adj[v]) for all v in P is O(|P|^2) or O(|P| * word_size).
        
        # Let's try a simpler bound first: |P|
        # If |C| + |P| <= best, prune.
        # This is very weak.
        
        # Let's implement the greedy coloring bound properly.
        # Colors assigned to vertices in P.
        # We process vertices in P in a specific order.
        # Let's use the order defined by 'order' list, but only those in P.
        
        # To speed up, we can precompute the list of vertices in P in 'order'.
        # But P changes.
        
        # Alternative: Use the "coloring" trick where we assign colors 1, 2, ...
        # and keep track of the max color.
        
        # This function is called at every node. It must be fast.
        # M is ~200.
        
        colors = {}
        max_c = 0
        
        # Iterate vertices in P
        # We can extract vertices from P
        p_temp = P
        while p_temp:
            # Pick a vertex. Let's pick the lowest set bit? Or highest degree?
            # Picking lowest set bit is fast.
            v = (p_temp & -p_temp).bit_length() - 1
            p_temp &= p_temp - 1
            
            # Determine available colors
            # A color c is available if no neighbor in P has color c.
            # This is expensive to check dynamically.
            
            # Simplified Greedy Coloring:
            # Assign color = 1 + max(color of neighbors in P)
            # This requires knowing colors of neighbors.
            
            # This is getting complex for pure Python speed.
            # Let's stick to |C| + |P| bound but with a very good vertex ordering and 
            # maybe a small local search improvement?
            
            # Actually, for N=200, |C|+|P| bound with good ordering (degree) often works 
            # if the graph is not too dense. The "compatible" graph H is likely dense 
            # (most sums are not squares). 
            # If H is dense, the Max Clique is large. 
            # If H is dense, |P| is a decent bound initially, but becomes loose.
            
            pass
            
        # Let's try a different approach: 
        # Use the standard Tomita algorithm logic with bitsets.
        # We will implement the coloring inside the recursion.
        
        return 0 # Placeholder

    # Let's implement a robust Tomita solver.
    
    def max_clique_search(P, C_size):
        nonlocal best_clique_size, best_clique_set
        
        if P == 0:
            if C_size > best_clique_size:
                best_clique_size = C_size
            return
        
        # Bound: C_size + P.bit_count()
        if C_size + P.bit_count() <= best_clique_size:
            return
            
        # Time check
        if time.time() > DEADLINE:
            return

        # Select vertex
        # Heuristic: Pick vertex with highest degree in P
        # To avoid O(|P|^2) scan every time, we can just pick the first one?
        # Or pre-sorted list?
        
        # Let's pick the vertex with the highest static degree among those in P.
        # We have 'order' sorted by static degree desc.
        # Find the first v in 'order' that is in P.
        
        v = -1
        for u in order:
            if (P >> u) & 1:
                v = u
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New P = P & adj[v]
        P_new = P & adj[v]
        max_clique_search(P_new, C_size + 1)
        
        # Branch 2: Exclude v
        P_excl = P & ~(1 << v)
        max_clique_search(P_excl, C_size)

    # Initialize
    # Start with all vertices
    # To make it faster, we can start with a greedy solution as a lower bound.
    
    def greedy_clique():
        P = (1 << M) - 1
        C = 0
        while P:
            # Pick vertex with max degree in P
            max_deg = -1
            v = -1
            p_temp = P
            while p_temp:
                u = (p_temp & -p_temp).bit_length() - 1
                p_temp &= p_temp - 1
                deg = (P & adj[u]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
            C |= (1 << v)
            P &= adj[v]
        return C.bit_count()
        
    init_clique = greedy_clique()
    best_clique_size = init_clique
    
    # Run search
    # We should order the initial P by degree to help the search?
    # The search function picks the highest static degree vertex available in P.
    
    all_vertices_mask = (1 << M) - 1
    max_clique_search(all_vertices_mask, 0)
    
    # Extract best clique
    # The search above didn't store the best set, only the size.
    # We need the set to return.
    # Let's modify the search to store the best set.
    
    # Re-implementing with set storage is safer.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    def search(P, C_mask, C_size):
        nonlocal best_clique_size, best_clique_mask
        
        if P == 0:
            if C_size > best_clique_size:
                best_clique_size = C_size
                best_clique_mask = C_mask
            return
            
        if C_size + P.bit_count() <= best_clique_size:
            return
            
        if time.time() > DEADLINE:
            return

        # Pick vertex v from P (highest static degree)
        v = -1
        for u in order:
            if (P >> u) & 1:
                v = u
                break
                
        if v == -1:
            return
            
        # Include v
        P_new = P & adj[v]
        search(P_new, C_mask | (1 << v), C_size + 1)
        
        # Exclude v
        P_excl = P & ~(1 << v)
        search(P_excl, C_mask, C_size)

    # Update lower bound with greedy
    greedy_mask = (1 << M) - 1
    current_C = 0
    while greedy_mask:
        max_deg = -1
        v = -1
        p_temp = greedy_mask
        while p_temp:
            u = (p_temp & -p_temp).bit_length() - 1
            p_temp &= p_temp - 1
            deg = (greedy_mask & adj[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
        current_C |= (1 << v)
        greedy_mask &= adj[v]
        
    best_clique_size = current_C.bit_count()
    best_clique_mask = current_C
    
    # Start search
    search((1 << M) - 1, 0, 0)
    
    # Map back to original values
    result = []
    for i in range(M):
        if (best_clique_mask >> i) & 1:
            result.append(valid_vertices[i])
            
    return sorted(result)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
