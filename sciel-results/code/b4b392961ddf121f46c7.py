import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n = 16
    NUM_COLORS = 3
    
    # Total possible edges
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Precompute all edge tuples
    all_edges = list(combinations(range(n), 2))
    
    # Helper: Check triangle-free condition for adding an edge
    # Returns True if adding (u,v) keeps triangle-free
    def can_add_edge(adj, u, v):
        # If u and v have a common neighbor, adding (u,v) creates a triangle
        common = adj[u] & adj[v]
        return common == 0
    
    # Helper: Exact 3-colorability check using backtracking
    # Uses bitmasks for speed
    def is_3_colorable(adj, vertex_order=None):
        colors = [-1] * n
        
        # Determine vertex ordering: highest degree first (with tie-breaking by index)
        if vertex_order is None:
            degrees = [bin(a).count('1') for a in adj]
            vertex_order = sorted(range(n), key=lambda x: (-degrees[x], x))
        
        # Precompute neighbor masks for each vertex in the ordering
        # Actually, just use the adj masks directly
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = vertex_order[idx]
            
            # Find forbidden colors
            forbidden = 0
            nb_mask = adj[v]
            
            # Iterate through set bits in nb_mask
            m = nb_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    forbidden |= (1 << colors[u])
                m -= lsb
            
            # Try colors 0, 1, 2
            for c in range(NUM_COLORS):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Helper: Get edge count from adjacency masks
    def edge_count(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2
    
    # Helper: Get list of edges from adjacency masks
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Helper: Try to add as many edges as possible greedily
    # Returns the final adjacency and whether it's 4-chromatic
    def greedy_maximize(adj, enforce_4_chromatic=True):
        # First, ensure it's 4-chromatic
        if enforce_4_chromatic:
            if is_3_colorable(adj):
                # Need to add edges to make it 4-chromatic
                # This is hard to do greedily.
                # If we start with a 4-chromatic core, this shouldn't happen.
                pass
        
        # Collect all potential edges that are not currently present
        # Sort by some heuristic? 
        # Maybe edges where u and v have many disjoint neighbors?
        # Or just random order?
        
        # Let's try random shuffling for diversity, or a deterministic "most constrained first"?
        # Actually, for maximum edge density, adding "easy" edges first might block "hard" ones?
        # No, if an edge can be added, it's always safe triangle-wise.
        # But it might prevent OTHER edges from being added later.
        # So we should prioritize edges that have the FEWEST conflicts?
        # Or edges that are part of fewest triangles?
        # Actually, let's just try all non-edges and add them in a random order.
        
        non_edges = []
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if can_add_edge(adj, u, v):
                # Check 3-colorability impact?
                # Adding an edge might make a 3-colorable graph 4-colorable.
                # If the graph is already 4-colorable, adding an edge keeps it 4-colorable.
                # If the graph is 3-colorable, adding an edge MIGHT make it 4-colorable.
                # We WANT it to be 4-colorable.
                # So:
                # 1. If current graph is 4-chromatic: Add if triangle-free.
                # 2. If current graph is 3-colorable: Add if triangle-free. Then check if it became 4-chromatic.
                #    If it's still 3-colorable, is that bad?
                #    Yes, because we need the final graph to be 4-chromatic.
                #    But if we add MORE edges, it might become 4-chromatic later.
                #    So we can add it tentatively.
                
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Optional: Check colorability?
                # Doing this every step is expensive (backtracking).
                # Let's only check at the end, or periodically?
                # If we check at the end, we might end up with a 3-colorable graph.
                # So we MUST ensure the final graph is 4-chromatic.
                
                # To be safe and efficient:
                # We assume we start with a 4-chromatic core.
                # Then any superset is 4-chromatic.
                # So we don't need to check 3-colorability during the greedy phase
                # IF we guarantee the core is 4-chromatic.
                
        return adj

    # Helper: Construct the Grötzsch graph core (11 vertices) + 5 isolated
    # Then fill the 5 new vertices optimally?
    # Actually, let's just use the Grötzsch graph on vertices 0-10.
    # Vertices 11-15 are initially isolated.
    # We can try to connect them to maximize edges.
    
    def create_grotzsch_core():
        adj = [0] * n
        
        # C5: 0-1-2-3-4-0
        adj[0] |= (1 << 1) | (1 << 4)
        adj[1] |= (1 << 0) | (1 << 2)
        adj[2] |= (1 << 1) | (1 << 3)
        adj[3] |= (1 << 2) | (1 << 4)
        adj[4] |= (1 << 3) | (1 << 0)
        
        # Copies:
        # 5 connects to 4, 0
        adj[5] |= (1 << 4) | (1 << 0)
        adj[4] |= (1 << 5)
        adj[0] |= (1 << 5)
        
        # 6 connects to 0, 1
        adj[6] |= (1 << 0) | (1 << 1)
        adj[0] |= (1 << 6)
        adj[1] |= (1 << 6)
        
        # 7 connects to 1, 2
        adj[7] |= (1 << 1) | (1 << 2)
        adj[1] |= (1 << 7)
        adj[2] |= (1 << 7)
        
        # 8 connects to 2, 3
        adj[8] |= (1 << 2) | (1 << 3)
        adj[2] |= (1 << 8)
        adj[3] |= (1 << 8)
        
        # 9 connects to 3, 4
        adj[9] |= (1 << 3) | (1 << 4)
        adj[3] |= (1 << 9)
        adj[4] |= (1 << 9)
        
        # 10 connects to 5, 6, 7, 8, 9
        adj[10] |= (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
        adj[5] |= (1 << 10)
        adj[6] |= (1 << 10)
        adj[7] |= (1 << 10)
        adj[8] |= (1 << 10)
        adj[9] |= (1 << 10)
        
        # Vertices 11-15 are isolated
        return adj

    # Main search loop
    best_adj = None
    best_count = 0
    best_edges = None
    
    # Restart loop
    num_restarts = 10000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Create core
        adj = create_grotzsch_core()
        
        # 2. Randomly initialize connections for vertices 11-15?
        # Or just start with isolated and let greedy fill?
        # Let's try adding random edges for 11-15 to diversify.
        # But we must maintain triangle-free.
        # Since 11-15 are isolated, adding edges from them to 0-10 is always safe
        # UNLESS they create triangles with each other? No, they are isolated from each other initially.
        # But if we add (11, 0) and (11, 1), and (0,1) exists, we have a triangle!
        # So we must check.
        
        # Let's just do pure greedy maximization on the core.
        # But to avoid always getting the same result, we can randomize the order of edge addition.
        
        # Greedy fill
        adj = greedy_maximize(adj)
        
        # Verify 4-chromaticity (should be true since core is 4-chromatic)
        if not is_3_colorable(adj):
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = [a | 0 for a in adj] # Copy
                best_edges = get_edges(adj)
        
        # Optional: Local Search / Swap
        # Try to swap edges to increase count?
        # Since we did a full greedy pass, it's likely maximal w.r.t single edge additions.
        # But a "stuck" state might require removing an edge to add two others.
        # Let's implement a simple swap search if we have time.
        
        if time.time() < deadline - 5.0:
            # Try swaps
            improved = True
            swap_count = 0
            while improved and swap_count < 50 and time.time() < deadline - 2.0:
                improved = False
                swap_count += 1
                
                # Collect all non-edges
                non_edges = []
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
                
                if not non_edges:
                    break
                
                # Try to remove one edge and add two?
                # Or just try to find a non-edge that can be added by removing ONE edge?
                # If we remove edge (a,b), can we add (c,d) and (e,f)?
                # This is complex.
                
                # Simpler: Random restarts with different randomization in greedy.
                # The greedy function already shuffles.
                # Let's just rely on the restarts.
                pass
        
    # If no valid graph found (should not happen), fallback to Grötzsch core
    if best_adj is None:
        adj = create_grotzsch_core()
        # Quick check
        if not is_3_colorable(adj):
            best_adj = adj
            best_edges = get_edges(adj)
        else:
            # Critical failure: Grötzsch core is 3-colorable?
            # This shouldn't happen. Return empty? No, must be valid.
            # Let's return the core anyway, assuming the checker is strict.
            # But if it's 3-colorable, score is 0.
            # Let's hardcode a known 4-chromatic triangle-free graph just in case.
            # Actually, the Grötzsch graph IS 4-chromatic.
            # If my code says it's 3-colorable, my code is buggy.
            # Let's force return the core edges.
            best_edges = get_edges(adj)

    return best_edges

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Found {len(edges)} edges")
    
    # Verify
    n = 16
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check triangle-free
    tf = True
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                if adj[u] & adj[v]:
                    tf = False
                    print(f"Triangle found at {u},{v}")
    print(f"Triangle-free: {tf}")
    
    # Check 3-colorable
    col = is_3_colorable(adj)
    print(f"3-Colorable: {col}")
    if col:
        print("WARNING: Graph is 3-colorable!")
    else:
        print("Success: Graph is not 3-colorable (4-chromatic)")
