"""Rewritten program for R(3,10) triangle-free graph construction."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds; leave safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def remaining_time():
        return DEADLINE - elapsed()

    # Precomputed Andrasfai graph And(9) on 26 vertices
    # Vertices 0..25, edge (i,j) if (j-i) mod 26 is in {1, 2} (mod 3 == 1 or 2, but not 0)
    # Actually And(9): vertices 0..8 repeated 3 times? No.
    # Andrasfai graph And(k) on 3k vertices. And(9) has 27 vertices.
    # Wait, let me recheck. And(3) has 9 vertices. And(9) has 27 vertices.
    # The Andrasfai graph And(k) is the circulant graph on 3k vertices where
    # vertex i is connected to i+1, i+2, ..., i+k (mod 3k).
    # For k=9: 27 vertices, each connected to 18 others (9 on each side? No, k on each side means degree 2k? No.)
    # Standard definition: And(k) is on 3k vertices. Vertex i connected to i+j mod 3k for j=1..k.
    # So degree is 2k? No, j=1..k gives k neighbors in one direction, and by symmetry k in the other?
    # Actually, it's a circulant. i~j iff (j-i) mod 3k is in {1, ..., k} or {3k-k, ..., 3k-1}.
    # So degree is 2k.
    # For k=9: 27 vertices, degree 18.
    # Is it triangle-free? 
    # Triangle would be i, i+1, i+2. (i,i+1) edge, (i+1,i+2) edge, (i,i+2) edge.
    # Yes, that's a triangle! So And(k) is NOT triangle-free for k>=2.
    # 
    # Let me reconsider. The triangle-free Andrasfai graphs are different.
    # Actually, I recall now: the Andrasfai-Erdos-Sos theorem is about triangle-free graphs.
    # The specific triangle-free constructions I should use:
    # 
    # 1. The Clebsch graph: 16 vertices, triangle-free, alpha=5.
    # 2. The Schrijver graph?
    # 3. Known R(3,10) lower bound: 39.
    #
    # Let me think about what graph on 39 vertices is triangle-free with alpha < 10.
    # 
    # Actually, a well-known construction uses the polarity graph or related structures.
    # 
    # Let me try a different approach: use a random triangle-free graph with controlled edge density,
    # then use local search to minimize alpha.
    #
    # Strategy:
    # 1. Start with a random triangle-free graph on n vertices.
    # 2. Use local search: try to add edges (while maintaining triangle-free) to reduce alpha.
    # 3. Also try to add vertices one at a time.
    #
    # Key insight: For a triangle-free graph, alpha >= n / (1 + sqrt(4*E/n + 1)) approximately.
    # To have alpha < 10 with n=39, we need enough edges.
    # By Turan, max edges in triangle-free graph on n vertices is n^2/4.
    # For n=39, max edges ~ 380.
    # 
    # Let me try a systematic approach:
    # - Start with known small triangle-free graphs with good alpha.
    # - Use a "blow-up" or "product" construction to get larger graphs.
    #
    # Actually, let me just do a well-optimized local search from random starts.

    def build_adj(n, edges):
        """Build adjacency sets from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free_adj(n, adj):
        """Check if graph is triangle-free using adjacency sets."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[i]:
                        if k > j and k in adj[j]:
                            return False
        return True

    def max_independent_set_size(n, adj):
        """Exact branch and bound for maximum independent set size."""
        best = [0]
        
        def branch(remaining, size):
            if elapsed() > DEADLINE - 3:
                return
            if not remaining:
                if size > best[0]:
                    best[0] = size
                return
            if size + len(remaining) <= best[0]:
                return
            # Pick vertex with max degree in remaining subgraph
            r = list(remaining)
            # Use a heuristic: pick vertex with max number of neighbors in remaining
            v = max(r, key=lambda x: len(adj[x] & remaining))
            # Branch 1: include v
            new_remaining = remaining - {v} - adj[v]
            branch(new_remaining, size + 1)
            # Branch 2: exclude v
            new_remaining2 = remaining - {v}
            branch(new_remaining2, size)

        branch(set(range(n)), 0)
        return best[0]

    def check_valid(n, edges):
        """Check if graph is triangle-free and has alpha < 10."""
        if elapsed() > DEADLINE - 5:
            return False
        adj = build_adj(n, edges)
        if not is_triangle_free_adj(n, adj):
            return False
        alpha = max_independent_set_size(n, adj)
        return alpha < 10

    def generate_random_triangle_free(n, p=0.25, seed=42):
        """Generate a random triangle-free graph."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        # Process vertices in random order
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            for j in order:
                if j <= i:
                    continue
                # Check if adding edge (i,j) creates a triangle
                can_add = True
                common = adj[i] & adj[j]
                if common:
                    can_add = False
                if can_add and rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        return n, edges, adj

    def local_search(n, edges, adj, max_iter=5000):
        """Local search to minimize alpha while maintaining triangle-free."""
        best_alpha = max_independent_set_size(n, adj)
        if best_alpha < 10:
            return n, edges, adj, best_alpha
        
        for it in range(max_iter):
            if elapsed() > DEADLINE - 10:
                break
            # Try to add a random edge
            i, j = random.sample(range(n), 2)
            if j in adj[i]:
                continue
            # Check if adding (i,j) creates a triangle
            if adj[i] & adj[j]:
                continue
            # Tentatively add edge
            adj[i].add(j)
            adj[j].add(i)
            edges.append([i, j])
            
            new_alpha = max_independent_set_size(n, adj)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                if best_alpha < 10:
                    break
            else:
                # Undo
                adj[i].discard(j)
                adj[j].discard(i)
                edges.pop()
        
        return n, edges, adj, best_alpha

    def try_add_vertex(n, edges, adj):
        """Try to add a new vertex connected to a subset of existing vertices."""
        if elapsed() > DEADLINE - 15:
            return None
        new_v = n
        # Find a valid connection pattern
        # The new vertex can connect to any independent set in the current graph
        # (to avoid creating triangles)
        # We want to choose connections that minimize the increase in alpha
        
        # Get all maximal independent sets of the current graph (heuristic)
        # Or just try random subsets
        
        best_result = None
        for attempt in range(200):
            if elapsed() > DEADLINE - 15:
                break
            # Random subset of vertices to connect to
            subset_size = random.randint(0, n)
            if subset_size == 0:
                # New vertex is isolated - this increases alpha by 1
                continue
            subset = set(random.sample(range(n), min(subset_size, n)))
            # Check if subset is independent (no edges within subset)
            valid = True
            for u in subset:
                for v in subset:
                    if u < v and v in adj[u]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Build new graph
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(subset))
            for u in subset:
                new_adj[u].add(new_v)
            
            new_edges = edges + [[new_v, u] for u in subset]
            
            alpha = max_independent_set_size(new_n, new_adj)
            if alpha < 10:
                return new_n, new_edges, new_adj, alpha
        
        return None

    best_n = 0
    best_edges = []
    best_adj = None
    best_alpha = 100

    # Try multiple random starts with local search
    for seed in range(200):
        if elapsed() > DEADLINE - 20:
            break
        
        # Try different target sizes
        for target_n in [26, 30, 34, 38, 40]:
            if elapsed() > DEADLINE - 15:
                break
            
            n, edges, adj = generate_random_triangle_free(target_n, p=0.3, seed=seed * 100 + target_n)
            alpha = max_independent_set_size(n, adj)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_n = n
                best_edges = edges
                best_adj = adj
            
            if alpha < 10:
                # We have a valid graph, try to extend
                break
            
            # Local search to reduce alpha
            n, edges, adj, alpha = local_search(n, edges, adj, max_iter=1000)
            if alpha < best_alpha:
                best_alpha = alpha
                best_n = n
                best_edges = edges
                best_adj = adj
            
            if alpha < 10:
                break
    
    # Try to extend the best graph found
    if best_adj is not None and best_alpha >= 10:
        for _ in range(50):
            if elapsed() > DEADLINE - 10:
                break
            result = try_add_vertex(best_n, best_edges, best_adj)
            if result is not None:
                best_n, best_edges, best_adj, best_alpha = result
                if best_alpha < 10:
                    break

    # Final check: if we have a valid graph, return it
    if best_alpha < 10 and best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: return the Andrasfai-like graph or a known small valid graph
    # Let me construct a known valid graph: the Clebsch graph has 16 vertices, alpha=5
    # But we want larger. Let me try to return the best we found even if alpha >= 10?
    # No, the task requires alpha < 10.
    
    # Let me try a specific known construction:
    # The graph from the R(3,10) lower bound of 39.
    # I'll try to construct it using a polarity graph or similar.
    
    # Actually, let me try a different approach: use the known 39-vertex construction.
    # One known construction uses the incidence graph of a projective plane or similar.
    
    # For now, let me try harder with local search from better initial points.
    for seed in range(500):
        if elapsed() > DEADLINE - 5:
            break
        for target_n in [39, 40]:
            if elapsed() > DEADLINE - 5:
                break
            n, edges, adj = generate_random_triangle_free(target_n, p=0.35, seed=seed * 200 + target_n)
            alpha = max_independent_set_size(n, adj)
            if alpha < 10:
                return {'n': n, 'edges': edges}
            n, edges, adj, alpha = local_search(n, edges, adj, max_iter=500)
            if alpha < 10:
                return {'n': n, 'edges': edges}
    
    # Last resort: return a small valid graph
    # Clebsch graph: 16 vertices
    n = 16
    edges = []
    # Clebsch graph vertices are binary strings of length 4 with even parity
    # Edge if Hamming distance is 2
    vertices = []
    for i in range(16):
        bits = bin(i)[2:].zfill(4)
        if sum(int(b) for b in bits) % 2 == 0:
            vertices.append(i)
    # Actually, let me just use a known edge list for a small valid graph
    # Or return the best we have
    if best_n > 0 and best_alpha < 10:
        return {'n': best_n, 'edges': best_edges}
    
    # Return a minimal valid graph (triangle-free with alpha < 10)
    # A complete bipartite graph K_{9,9} is triangle-free with alpha = 9
    n = 18
    edges = [[i, j] for i in range(9) for j in range(9, 18)]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
