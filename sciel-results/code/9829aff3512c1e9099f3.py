import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 150s limit
    
    # We want to find a C4-free graph with alpha < 11, maximizing n
    # Best known: 39 vertices
    
    # Strategy: Use a randomized greedy construction with backtracking
    # Key insight: C4-free graphs have strong structure. We can use
    # the fact that in a C4-free graph, any two vertices share at most 1 common neighbor.
    
    # Approach: Build graph by adding vertices one at a time
    # For each new vertex, we choose a subset of existing vertices to connect to
    # such that no C4 is created and the independence number stays below 11
    
    # Let's use a different approach: start from known C4-free constructions
    # and try to extend them.
    
    # Actually, let's use a more systematic approach:
    # 1. Use a finite geometry / combinatorial design to get a C4-free graph
    # 2. Verify alpha < 11
    # 3. Try to add vertices
    
    # For C4-free graphs, one powerful construction is from finite projective planes.
    # The incidence graph of a projective plane of order q is C4-free.
    # But that gives bipartite graphs.
    
    # Let's try: polar graph of a quadratic form, or use known small C4-free graphs.
    
    # Alternative: use the complement of a linear hypergraph.
    # A linear hypergraph (any two edges share at most 1 vertex) gives a C4-free graph
    # where vertices of the hypergraph are edges of the graph, etc.
    
    # Let me try a direct construction approach:
    # We'll build the graph incrementally, maintaining:
    # - adj: adjacency as bitmasks (Python ints for up to ~64 vertices)
    # - For C4-free: when adding edge (u,v), check that N(u) & N(v) = 0 (no common neighbor)
    #   Actually no: we need that adding edge (u,v) doesn't create a C4.
    #   A C4 is u-w-v-x-u where w and x are common neighbors of u and v.
    #   So we need |N(u) ∩ N(v)| <= 1 for all pairs (u,v) that are NOT adjacent.
    #   Wait, that's the condition for the whole graph. When adding a new vertex, we need
    #   to check all pairs.
    
    # Let me use a different strategy:
    # Start with a random graph, then iteratively remove edges to eliminate C4s,
    # while trying to keep alpha small.
    
    # Actually, the most promising approach for this time budget:
    # Use a known 39-vertex construction and verify it.
    
    # The best known R(C4,K11) > 39 construction... Let me think about what's known.
    # 
    # Actually, let me try a constructive approach:
    # Use the complement of a linear 2-(v, k, λ) design or similar.
    
    # Let's try: the line graph of a linear hypergraph.
    # If H is a linear hypergraph on v vertices with edges of size k,
    # then the "intersection graph" (vertices = edges of H, edge if they intersect)
    # is C4-free iff H is linear (any two edges share at most 1 vertex).
    # Wait, that's not quite right either.
    
    # Let me try a simpler approach: greedy random construction with verification.
    
    def is_c4_free(adj, n):
        """Check if graph with adjacency bitmasks adj (list of ints) is C4-free."""
        for i in range(n):
            for j in range(i+1, n):
                # Count common neighbors of i and j
                common = adj[i] & adj[j]
                if (common & (common - 1)) != 0:  # more than 1 bit set
                    return False
        return True
    
    def independence_number(adj, n):
        """Compute alpha using branch and bound with bitmask."""
        # Use a greedy upper bound + branch and bound
        best = [0]
        
        def branch(candidates, size, start):
            if time.time() > deadline:
                return
            if size + candidates.bit_count() <= best[0]:
                return
            if candidates == 0:
                if size > best[0]:
                    best[0] = size
                return
            
            # Greedy: pick a vertex with minimum degree in candidates
            # For speed, just pick the lowest set bit
            # Better: pick vertex that appears in most edges among candidates
            # For now, use a simple heuristic
            
            # Get first vertex in candidates
            v = (candidates & (-candidates)).bit_length() - 1
            candidates_without_v = candidates & ~(1 << v)
            
            # Option 1: include v
            branch(candidates_without_v & ~adj[v], size + 1, start)
            # Option 2: exclude v
            branch(candidates_without_v, size, start)
        
        all_vertices = (1 << n) - 1
        branch(all_vertices, 0, 0)
        return best[0]
    
    # For speed, let's use a faster alpha check
    def alpha_less_than_11(adj, n):
        """Return True if independence number is less than 11 (i.e., <= 10)."""
        # Try to find an independent set of size 11
        # If we can't, return True
        found = [False]
        
        def find_ind_set(candidates, size):
            if found[0] or time.time() > deadline:
                return
            if size == 11:
                found[0] = True
                return
            if candidates == 0:
                return
            
            v = (candidates & (-candidates)).bit_length() - 1
            # Include v
            find_ind_set(candidates & ~adj[v] & ~(1 << v), size + 1)
            if found[0]:
                return
            # Exclude v
            find_ind_set(candidates & ~(1 << v), size)
        
        all_v = (1 << n) - 1
        find_ind_set(all_v, 0)
        return not found[0]
    
    # Now let's try to construct a large C4-free graph with alpha <= 10.
    # Strategy: random greedy vertex addition.
    
    def try_construction(seed, target_n=39):
        """Try to build a C4-free graph with alpha <= 10 on target_n vertices."""
        rng = random.Random(seed)
        adj = [0] * target_n
        edge_list = []
        
        # Start with no edges, add vertices
        # Actually, let's build edges greedily.
        # We'll try to make the graph as dense as possible while staying C4-free.
        
        # For a C4-free graph on n vertices, the maximum number of edges is O(n^{3/2}).
        # Specifically, ex(n, C4) ≈ n^{3/2}/2 for large n.
        
        # Let's use a random greedy edge addition:
        # Shuffle all possible edges, add each if it doesn't create a C4.
        
        all_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                return None
            # Check if adding edge (u,v) creates a C4
            # A C4 would be: u - x - v - y - u where x, y are distinct common neighbors
            # of u and v. But wait, u and v are not yet adjacent, so we need to check
            # if |N(u) ∩ N(v)| >= 2.
            common = adj[u] & adj[v]
            if common.bit_count() >= 2:
                continue  # Would create C4
            
            # Also need to check: does adding (u,v) create a C4 through other paths?
            # Actually, the C4 condition is: no two vertices have 2+ common neighbors.
            # Adding edge (u,v) means u and v become adjacent. The C4 constraint is
            # about common neighbors, not about being adjacent. So we just need
            # |N(u) ∩ N(v)| <= 1 after addition. Since u and v weren't adjacent before,
            # N(u) doesn't include v and N(v) doesn't include u. Adding the edge doesn't
            # change N(u) or N(v) in terms of other vertices. So the check is just
            # |N(u) ∩ N(v)| <= 1 before adding.
            
            # Wait, that's correct. The C4-free condition is equivalent to:
            # for every pair of vertices (not necessarily adjacent), they have at most
            # 1 common neighbor. Adding an edge between u and v doesn't affect the
            # common neighbors of any other pair, and for the pair (u,v), their common
            # neighbors don't change when we add the edge (u,v) itself.
            
            # So we just need |N(u) ∩ N(v)| <= 1.
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_list.append([u, v])
        
        # Check alpha
        if not alpha_less_than_11(adj, target_n):
            return None
        
        return {'n': target_n, 'edges': edge_list}
    
    best_result = None
    
    # Try multiple seeds
    for seed in range(100000):
        if time.time() > deadline:
            break
        
        result = try_construction(seed, 39)
        if result is not None:
            best_result = result
            break
    
    # If we couldn't find a 39-vertex graph, try smaller
    if best_result is None:
        for target_n in range(38, 29, -1):
            if time.time() > deadline:
                break
            for seed in range(10000):
                if time.time() > deadline:
                    break
                result = try_construction(seed, target_n)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break
    
    # Fallback: return the seed solution
    if best_result is None:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_result = {'n': 30, 'edges': edges}
    
    return best_result
