import time
import random

def construct():
    # Deadline guard: 150s wall-clock limit, use 130s for search, 20s margin
    start_time = time.time()
    deadline = start_time + 130

    # 1. Construct the Clebsch Graph (Triangle-free, 16 vertices, 40 edges, chi=4)
    # This is the folded 5-cube.
    # Vertices are 5-bit strings. Two vertices u, v are connected if they differ in exactly 1 bit,
    # considering the folded structure where x and x^31 are identified.
    
    # Map 32 nodes to 16 classes
    # Class of x is min(x, x^31)
    def get_class(x):
        return min(x, x ^ 31)
    
    class_to_id = {}
    id_to_class = []
    for i in range(32):
        c = get_class(i)
        if c not in class_to_id:
            class_to_id[c] = len(class_to_id)
            id_to_class.append(c)
            
    node_to_id = {}
    for i in range(32):
        node_to_id[i] = class_to_id[get_class(i)]
        
    edges_set = set()
    for i in range(32):
        u_id = node_to_id[i]
        for k in range(5):
            j = i ^ (1 << k)
            v_id = node_to_id[j]
            if u_id < v_id:
                edges_set.add((u_id, v_id))
                
    initial_edges = [list(e) for e in edges_set]
    initial_count = len(initial_edges)
    
    # Represent graph as list of 16-bit integers (bitmasks)
    def build_adj(edge_list):
        adj = [0] * 16
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    adj = build_adj(initial_edges)
    
    # 2. Verification Functions
    def is_triangle_free(adj):
        for u in range(16):
            # For each neighbor v of u, check if u and v share any other common neighbor w
            # If (u,v) is an edge, and adj[u] & adj[v] != 0, then there exists w connected to both u and v.
            # Since u-v is an edge, u-v-w is a triangle.
            # Note: adj[u] & adj[v] includes bits for common neighbors.
            # If (u,v) is an edge, v is in adj[u] and u is in adj[v].
            # The intersection adj[u] & adj[v] will contain bits for nodes connected to both.
            # Specifically, if w is a common neighbor, bit w is set.
            # So if (adj[u] & adj[v]) != 0, it implies a triangle?
            # Wait. If u and v are connected, and they have a common neighbor w, then (u,v), (u,w), (v,w) form a triangle.
            # The condition for triangle-free is that for every edge (u,v), N(u) \ {v} and N(v) \ {u} are disjoint.
            # Equivalently, (adj[u] & adj[v]) should be 0?
            # Let's check: if (u,v) is an edge, adj[u] has bit v, adj[v] has bit u.
            # If they share a neighbor w, adj[u] has bit w, adj[v] has bit w.
            # So adj[u] & adj[v] will have bit w set.
            # Thus, if (adj[u] & adj[v]) != 0, there is a triangle.
            # Is it possible for adj[u] & adj[v] to be non-zero without a triangle?
            # Only if u is in adj[v] and v is in adj[u]? No, that's the edge itself.
            # But the intersection includes ALL common neighbors.
            # If the only common "neighbor" in the bit representation is each other?
            # No, u is not in adj[u] (simple graph, no self loops).
            # So if (u,v) is an edge, adj[u] has v, adj[v] has u.
            # The bit for v in adj[u] is 1. The bit for v in adj[v] is 0 (no self loop).
            # The bit for u in adj[u] is 0. The bit for u in adj[v] is 1.
            # So u and v do not contribute to the intersection adj[u] & adj[v].
            # Therefore, if (adj[u] & adj[v]) != 0, it MUST be due to a third vertex w.
            # So the check is correct.
            for v in range(16):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_not_3_colorable(adj):
        # Backtracking 3-coloring with MRV (Minimum Remaining Values) heuristic
        # This is significantly faster than naive ordering, especially for dense graphs.
        
        # Precompute neighbors list for faster access? Or just use bitmasks.
        # We'll use a recursive backtracker.
        
        # To speed up, we can order vertices by degree descending initially,
        # but dynamic MRV is better.
        
        # State: colors (list of -1, 0, 1, 2), uncolored set (bitmask)
        
        # For simplicity and speed in Python, a simple recursive backtracker with
        # forward checking (checking if any neighbor has the color) is usually enough for 16 nodes.
        
        colors = [-1] * 16
        
        # Helper to get available colors for vertex v
        def get_available(v):
            avail = [0, 1, 2]
            # Check neighbors
            mask = adj[v]
            for c in range(3):
                # If any neighbor has color c, c is unavailable
                # We need to check if any neighbor j with colors[j]==c exists
                # Optimization: maintain counts of colors in neighbors?
                # For 16 nodes, simple loop is fine.
                for j in range(16):
                    if mask & (1 << j):
                        if colors[j] == c:
                            avail.remove(c)
                            break
            return avail

        def backtrack(unassigned_mask):
            if unassigned_mask == 0:
                return True
            
            # Select vertex with minimum available colors (MRV)
            # Iterate through bits in unassigned_mask
            best_v = -1
            best_avail = None
            min_count = 4
            
            m = unassigned_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                avail = get_available(v)
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_avail = avail
                    if min_count == 0:
                        return False # Pruning: no colors available
            
            # Assign best_v
            for c in best_avail:
                colors[best_v] = c
                if backtrack(unassigned_mask & ~(1 << best_v)):
                    return True
                colors[best_v] = -1
            
            return False

        return not backtrack((1 << 16) - 1)

    # Check initial graph
    if not is_triangle_free(adj) or not is_not_3_colorable(adj):
        # Fallback to Groetzsch if Clebsch construction failed (should not happen)
        # Groetzsch graph: 11 vertices, 20 edges, chi=4, triangle-free.
        # Vertices 0..10. 11..15 isolated.
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]

    best_edges = initial_edges
    best_adj = adj[:]
    best_count = initial_count

    # 3. Structural Search: Random Perturbation + Greedy Maximization
    # Since the Clebsch graph is likely optimal (40 edges), we try to find a graph with > 40 edges.
    # Strategy:
    # 1. Start with a random subset of edges or a perturbed Clebsch graph.
    # 2. Ensure it is not 3-colorable (if it becomes 3-colorable, we might need to keep structure that forces chi>=4).
    #    Actually, removing edges can make a graph 3-colorable.
    #    Adding edges can create triangles.
    #    The constraint "not 3-colorable" is the hard one.
    #    
    # Improved Strategy:
    # Start with the Clebsch graph.
    # Try to swap edges: Remove a set of edges, add a set of edges.
    # Or: Start from scratch with a random triangle-free graph, then greedily add edges while maintaining non-3-colorability?
    # Checking non-3-colorability is expensive.
    
    # Better: Use the property that the Clebsch graph is regular and highly symmetric.
    # Try random edge swaps (2-opt): Remove 2 edges, add 2 edges.
    # Check if still triangle-free and not 3-colorable.
    
    # Let's implement a simple local search.
    # We will try many random restarts.
    
    def count_edges(adj):
        c = 0
        for i in range(16):
            c += bin(adj[i]).count('1')
        return c // 2

    def edges_from_adj(adj):
        e = []
        for u in range(16):
            for v in range(u+1, 16):
                if adj[u] & (1 << v):
                    e.append([u, v])
        return e

    # Local Search Loop
    # We use a time-bounded loop.
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(16) for v in range(u+1, 16)]
    
    # To speed up triangle check during greedy add:
    # When adding (u,v), check if N(u) and N(v) have common neighbors.
    # common = adj[u] & adj[v]
    # If common == 0, safe to add (regarding triangles).
    
    # To speed up 3-coloring check:
    # We only check at the end of a perturbation step.
    
    attempts = 0
    max_attempts = 50 # Limit attempts to avoid infinite loop if time is short
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Start from best_adj or a random perturbation of it
        current_adj = best_adj[:]
        
        # Randomly remove 2-4 edges
        current_edges = edges_from_adj(current_adj)
        if len(current_edges) < 2:
            continue
            
        for _ in range(random.randint(2, 4)):
            if not current_edges: break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges.pop(idx)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
        # Greedily add edges
        # Get list of non-edges
        non_edges = [(u, v) for u, v in all_edges if not (current_adj[u] & (1 << v))]
        
        # Shuffle non-edges to add randomness
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 5: break
            
            # Check triangle constraint
            common = current_adj[u] & current_adj[v]
            if common == 0:
                # Safe to add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        # Check validity
        if is_triangle_free(current_adj):
            if is_not_3_colorable(current_adj):
                count = count_edges(current_adj)
                if count > best_count:
                    best_count = count
                    best_adj = current_adj
                    best_edges = edges_from_adj(current_adj)
                    
                    # If we found a better graph, we can increase attempts or restart more aggressively
                    # But we are bounded by time.
                    
    # Final check
    if best_adj is not adj:
        return edges_from_adj(best_adj)
    else:
        return best_edges
