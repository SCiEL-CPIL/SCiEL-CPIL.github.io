import random
import time
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

def _is_k4_free(adj_bits):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        na = adj_bits[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj_bits[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                # If there's any common neighbor d > c, we have a K4
                # Actually we need any common neighbor d != a,b,c
                # common_abc already excludes a,b,c since they're not in adj_bits of each other in a simple graph?
                # Wait, adj_bits[a] does not contain a.
                # So common_abc contains vertices connected to a, b, AND c.
                # If common_abc is non-zero, there exists a vertex d connected to a,b,c.
                # That forms a K4 {a,b,c,d}.
                if common_abc:
                    return False
    return True

def _largest_triangle_free(adj_bits):
    """Find largest triangle-free induced subgraph size using bitmask enumeration."""
    
    def is_triangle_free(mask):
        # A subset S is triangle-free iff for every edge (u,v) in S, N(u) & N(v) & S == 0
        # Or: no three vertices u,v,w in S form a triangle.
        # Efficient check:
        # For each u in S, let N_S(u) = adj[u] & mask.
        # If N_S(u) contains an edge, then there's a triangle involving u.
        # So we check if N_S(u) is an independent set.
        m = mask
        while m:
            low_bit = m & (-m)
            u = low_bit.bit_length() - 1
            m ^= low_bit
            nu = adj_bits[u] & mask
            # Check if nu is an independent set
            nn = nu
            while nn:
                low_bit2 = nn & (-nn)
                v = low_bit2.bit_length() - 1
                nn ^= low_bit2
                if adj_bits[v] & nu:
                    return False
        return True
    
    # Check sizes from N down to 1
    for size in range(N, 0, -1):
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_triangle_free(mask):
                return size
    return 0

def _bits_to_edges(adj_bits):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # --- Phase 1: Tripartite Construction (Inspiration 3/Strategy) ---
    # We model the graph as a tripartite graph with parts A, B, C.
    # To be K4-free, we need to ensure no K4 exists.
    # A K4 in a tripartite graph must have vertices distributed among the parts.
    # Possible distributions: 2-1-1 or 1-1-2 or 1-2-1 or 1-1-1-1 (impossible with 3 parts) or 3-1-0 (impossible for K4 as no edges within part)
    # Actually, if the graph is strictly tripartite (no intra-part edges), a K4 requires at least 2 vertices in some part?
    # No, a K4 has 4 vertices. If parts are A, B, C.
    # If we pick 2 from A, 1 from B, 1 from C.
    # For them to form a K4, the 2 in A must be connected to each other? No, no intra-part edges.
    # So a K4 cannot have 2 vertices from the same part if there are no intra-part edges.
    # Therefore, a strictly tripartite graph is K4-free if and only if it does not contain a K_{2,2,2}? No.
    # A K4 in a tripartite graph with parts A, B, C:
    # Vertices v1, v2, v3, v4.
    # By Pigeonhole Principle, if we have 4 vertices and 3 parts, at least one part has >= 2 vertices.
    # Let's say v1, v2 in A.
    # For v1, v2, v3, v4 to form a K4, v1 must be connected to v2.
    # But v1, v2 are in A, so they are NOT connected (strict tripartite).
    # Thus, **any strictly tripartite graph is K4-free**.
    
    # So, we can restrict our search to strictly tripartite graphs with parts of size 6, 6, 6.
    # This drastically reduces the search space and guarantees K4-freeness without checking.
    # We just need to minimize the largest triangle-free induced subgraph.
    
    # Let's try to construct such a graph.
    # Parts: A=[0..5], B=[6..11], C=[12..17]
    # Edges only between A-B, B-C, C-A.
    
    def tripartite_construct():
        adj = [0] * N
        # Initialize with complete tripartite K6,6,6
        # This has L=12 (e.g., A union B is independent? No, A and B are connected. 
        # Wait, in K6,6,6, A is an independent set, B is independent, C is independent.
        # A union B is NOT independent because there are edges between A and B.
        # Is A union B triangle-free?
        # Any triangle in A union B would need 3 vertices.
        # Since A and B are independent, any triangle must have vertices from both.
        # e.g., a1 in A, b1, b2 in B.
        # b1-b2 is not an edge. So no triangle with 2 in B.
        # Similarly no triangle with 2 in A.
        # So A union B is triangle-free. Size 12.
        
        # We want to break these large triangle-free sets.
        # We can add "cross" edges? No, we are restricted to tripartite structure to guarantee K4-free.
        # Wait, can we add intra-part edges?
        # If we add an edge within A, say (a1, a2), does it create a K4?
        # We need a1-a2, and two other vertices connected to both.
        # If we have a1-a2, and b1, b2 in B connected to both a1 and a2, then {a1, a2, b1, b2} is a K4.
        # So if we add intra-part edges, we must be careful to avoid K4s.
        
        # Strategy: Start with K6,6,6.
        # Remove edges to break triangles? No, K6,6,6 has no triangles.
        # We want to MINIMIZE the largest triangle-free subgraph.
        # Currently L=12.
        # To reduce L, we need to introduce triangles such that any large subset contains a triangle.
        # But adding edges creates triangles.
        # If we add an edge (a1, b1) - already there.
        # If we add an edge within A, (a1, a2):
        # This creates triangles with any b connected to both.
        # In K6,6,6, all b are connected to all a.
        # So (a1, a2) + any b forms a triangle.
        # So if we add many intra-part edges, we create many triangles.
        # But we must ensure no K4.
        
        # Let's try a hybrid approach:
        # Start with K6,6,6.
        # Add random intra-part edges while checking K4-freeness.
        
        best_adj = [0] * N
        # Initialize K6,6,6
        for i in range(6):
            for j in range(6, 12):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
            for j in range(12, 18):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
        for i in range(6, 12):
            for j in range(12, 18):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
        
        current_score = _largest_triangle_free(best_adj)
        
        # Try adding intra-part edges
        intra_edges = []
        for part in [range(0, 6), range(6, 12), range(12, 18)]:
            for u in part:
                for v in part:
                    if u < v:
                        intra_edges.append((u, v))
        
        rng.shuffle(intra_edges)
        
        for u, v in intra_edges:
            new_adj = best_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if _is_k4_free(new_adj):
                # Check if score improves
                # Optimization: only calculate score if we accept?
                # Or calculate and compare.
                # Since we are adding edges, L can only decrease or stay same?
                # Adding edges can create triangles, which might break triangle-free sets.
                # So L is non-increasing.
                best_adj = new_adj
                # We don't need to update current_score immediately if we just want the final graph
                # But for SA later, we might want it.
        
        return best_adj, _largest_triangle_free(best_adj)

    best_adj, best_score = tripartite_construct()
    
    if time.time() > deadline:
        return _bits_to_edges(best_adj)

    # --- Phase 2: Simulated Annealing ---
    # We can now perform SA on the general graph space, starting from this tripartite-enhanced graph.
    # This allows us to potentially break the tripartite structure if it leads to better scores,
    # while maintaining K4-freeness.
    
    current_adj = best_adj[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    step = 0
    last_improvement = 0
    
    while time.time() < deadline:
        step += 1
        
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add edge
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    last_improvement = step
        elif move_type < 0.6:
            # Remove edge
            existing_edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            if not existing_edges:
                continue
            u, v = rng.choice(existing_edges)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    last_improvement = step
        else:
            # Swap
            existing_edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not existing_edges or not non_edges:
                continue
            u1, v1 = rng.choice(existing_edges)
            u2, v2 = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if not _is_k4_free(new_adj):
                continue
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    last_improvement = step
        
        T = max(T_min, T * cooling)
        
        # Reheat if stuck
        if step - last_improvement > 5000:
            T = max(T, 3.0)
            last_improvement = step

    return _bits_to_edges(best_adj)
