"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _greedy_independent_set(adj, n):
    """Greedy maximal independent set. Returns size."""
    remaining = set(range(n))
    count = 0
    while remaining:
        # pick vertex with min degree in remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in adj[v] if u in remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        count += 1
        remaining.discard(best_v)
        for u in adj[best_v]:
            remaining.discard(u)
    return count

def _exact_max_independent_set(adj, n, timeout=5.0):
    """Exact maximum independent set via branch and bound. Returns size."""
    start = time.time()
    # Use complement: max independent set in G = max clique in complement
    # But we'll do branch and bound on independent set directly
    best = [0]
    # Sort vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    
    # Build adjacency as sets for fast lookup
    adj_sets = [set(adj[v]) for v in range(n)]
    
    def backtrack(candidates, current_size):
        if time.time() - start > timeout:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best[0]:
            return
        # Pick a vertex to branch on
        v = candidates[0]
        # Branch 1: include v
        new_candidates = [u for u in candidates[1:] if u not in adj_sets[v]]
        backtrack(new_candidates, current_size + 1)
        # Branch 2: exclude v
        backtrack(candidates[1:], current_size)
    
    backtrack(order, 0)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free. Returns True if no C4."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _construct_greedy(n_target, seed, deadline):
    """Greedy construction: start with empty graph, add edges to reduce alpha."""
    random.seed(seed)
    adj = [set() for _ in range(n_target)]
    
    # Start with no edges, alpha = n_target
    # We want to add edges to reduce alpha below 11
    # But we must maintain C4-free
    
    # Strategy: iteratively add edges that most reduce a greedy independent set
    # while maintaining C4-free
    
    # Precompute which pairs can be added without creating C4
    # A pair (u,v) can be added if for all w, the number of common neighbors of u and w
    # (excluding v) plus the number of common neighbors of v and w (excluding u) doesn't
    # create a C4. Actually, adding edge (u,v) creates a C4 if there exist x,y such that
    # x is adjacent to both u and v, and y is adjacent to both u and v, and x != y.
    # Wait, no. Adding edge (u,v) creates a C4 if there is a path u-x-v-y-u or similar.
    # Actually, a C4 is a cycle of length 4. Adding edge (u,v) creates a C4 if
    # there exist distinct x, y such that u-x-v-y-u is a cycle, i.e., x is neighbor of u and v,
    # y is neighbor of u and v, and x-y is an edge. No wait.
    # 
    # A C4 in the new graph would be a 4-cycle. If we add edge (u,v), a new C4 could be
    # u-v-x-y-u where x is a common neighbor of u and v... no.
    # 
    # Let me think more carefully. A C4 is four distinct vertices a,b,c,d with edges ab, bc, cd, da.
    # If we add edge (u,v), a new C4 involving this edge would be u-v-x-y-u where:
    # - v-x is an edge
    # - x-y is an edge  
    # - y-u is an edge
    # - u-v is the new edge
    # So x must be a neighbor of v, y must be a neighbor of u, and x-y must be an edge,
    # and x != u, y != v, x != y.
    #
    # Actually simpler: adding edge (u,v) creates a C4 iff there exist distinct vertices x, y
    # such that x is adjacent to both u and v (in the old graph), and y is adjacent to both u and v
    # (in the old graph). Wait no, that's not right either.
    #
    # Let me reconsider. In the old graph, suppose u and v have common neighbors. If u and v
    # already have two or more common neighbors, then the graph already has a C4 (u-x-v-y-u
    # where x,y are common neighbors and x-y is an edge). But wait, we need x-y to be an edge.
    #
    # Actually, the C4-free condition is: no two vertices have two or more common neighbors.
    # So if we maintain that invariant, adding edge (u,v) is safe iff u and v have at most
    # one common neighbor in the current graph. Because if they have 0 or 1 common neighbor,
    # adding the edge doesn't create a C4. If they have 2+ common neighbors, the graph already
    # violates C4-free (but we're maintaining the invariant, so this won't happen).
    #
    # Wait, but adding edge (u,v) could also create a C4 that doesn't involve u and v as
    # adjacent in the C4. No, a C4 involving the new edge (u,v) must have u and v as
    # adjacent vertices in the cycle. So the C4 would be u-v-x-y-u or u-v-y-x-u.
    # For u-v-x-y-u: x is neighbor of v, y is neighbor of u, x-y is edge, x!=y, x!=u, y!=v.
    # This doesn't directly relate to common neighbors of u and v.
    #
    # Hmm, let me just use the definition: C4-free means no two vertices share 2+ common neighbors.
    # This is equivalent to no C4. So I just need to check that after adding (u,v), no pair
    # has 2+ common neighbors. The only pairs whose common neighbor count changes are pairs
    # involving u or v.
    
    # For efficiency, let's just check the invariant directly.
    
    def can_add_edge(u, v):
        if v in adj[u]:
            return False
        # Check if adding (u,v) would create a C4
        # The C4-free condition: for all pairs (a,b), |N(a) ∩ N(b)| <= 1
        # Adding (u,v) changes N(u) and N(v)
        # Pairs to check: (u, w) for all w != v, and (v, w) for all w != u, and (u,v) itself
        # For pair (u,v): |N(u) ∩ N(v)| must be <= 1. Currently |N(u) ∩ N(v)| = c.
        # After adding, N(u) gains v, N(v) gains u. But v is not in N(u)∩N(v) currently,
        # and u is not in N(u)∩N(v) currently. So |N(u)∩N(v)| stays c.
        # Wait, N(u)∩N(v) are vertices adjacent to both u and v. Adding edge (u,v) doesn't
        # change which vertices are adjacent to both u and v. So this is fine.
        #
        # For pair (u, w) where w != v: N(u) gains v. So N(u)∩N(w) may gain v if v is in N(w).
        # So we need: if v in N(w), then |N(u)∩N(w) \ {v}| <= 0, i.e., |N(u)∩N(w)| <= 1 currently
        # and v not in N(u)∩N(w) (which is true since v not in N(u)).
        # Actually, N(u)∩N(w) currently doesn't include v (since v not in N(u)). After adding,
        # N(u)∩N(w) may include v if v in N(w). So new |N(u)∩N(w)| = old |N(u)∩N(w)| + (1 if v in N(w) else 0).
        # We need this <= 1.
        #
        # Similarly for (v, w).
        
        for w in range(n_target):
            if w == u or w == v:
                continue
            # Check pair (u, w)
            if w in adj[v]:  # v is in N(w), so v will be in N(u)∩N(w) after adding
                common_uw = len(adj[u] & adj[w])
                if common_uw >= 1:
                    return False
            # Check pair (v, w)
            if w in adj[u]:  # u is in N(w), so u will be in N(v)∩N(w) after adding
                common_vw = len(adj[v] & adj[w])
                if common_vw >= 1:
                    return False
        return True
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy: repeatedly add the edge that most reduces a greedy independent set
    # But this is expensive. Let's try a different approach.
    
    # Alternative: Start with a known good structure and modify.
    # The seed has 30 vertices (10 triangles). We want to get to 39.
    
    # Let's try: start with 39 isolated vertices, then add edges greedily
    # to reduce alpha to 10.
    
    # Actually, let's use a different strategy:
    # 1. Start with n_target vertices
    # 2. Use a greedy algorithm to add edges while maintaining C4-free
    # 3. At each step, add the edge that reduces the greedy independent set size the most
    
    # For speed, use a fast heuristic for independent set size
    
    def fast_alpha():
        """Fast greedy independent set size."""
        remaining = list(range(n_target))
        count = 0
        while remaining:
            # Pick vertex with min degree in remaining
            best_idx = 0
            best_deg = float('inf')
            for i, v in enumerate(remaining):
                deg = sum(1 for u in adj[v] if u in remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_idx = i
                    if deg == 0:
                        break
            count += 1
            v = remaining.pop(best_idx)
            # Remove neighbors of v
            remaining = [u for u in remaining if u not in adj[v]]
        return count
    
    # Try multiple random orderings of edge additions
    best_result = None
    best_alpha = n_target
    
    start_time = time.time()
    
    for trial in range(50):
        if time.time() > deadline:
            break
        random.seed(seed + trial)
        adj = [set() for _ in range(n_target)]
        
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if can_add_edge(u, v):
                # Check if adding this edge reduces alpha
                old_alpha = fast_alpha()
                add_edge(u, v)
                new_alpha = fast_alpha()
                if new_alpha >= old_alpha:
                    # Revert
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        final_alpha = fast_alpha()
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_adj = [set(s) for s in adj]
        
        if best_alpha <= 10:
            break
    
    if best_alpha <= 10:
        edges = []
        for u in range(n_target):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n_target, 'edges': edges}
    
    return None

def _construct_polynomial(n_target, deadline):
    """Use a known polynomial construction or structured approach."""
    # The best known is 39 vertices. Let's try to construct it.
    # One approach: use a finite geometry or strongly regular graph.
    
    # Actually, let's try a simpler approach: use the greedy method but with
    # a better initialization.
    
    # Start with a random C4-free graph and try to reduce alpha.
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n_target)]
        random.seed(seed)
        
        # Add random edges while maintaining C4-free
        all_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if can add
            ok = True
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if w in adj[v] and len(adj[u] & adj[w]) >= 1:
                    ok = False
                    break
                if w in adj[u] and len(adj[v] & adj[w]) >= 1:
                    ok = False
                    break
            if ok:
                adj[u].add(v)
                adj[v].add(u)
        
        # Now try to reduce alpha by adding more edges
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Find a pair that can be added and reduces alpha
            all_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target) if j not in adj[i]]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            improved = False
            for u, v in all_edges:
                ok = True
                for w in range(n_target):
                    if w == u or w == v:
                        continue
                    if w in adj[v] and len(adj[u] & adj[w]) >= 1:
                        ok = False
                        break
                    if w in adj[u] and len(adj[v] & adj[w]) >= 1:
                        ok = False
                        break
                if ok:
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
                    break
            
            if not improved:
                break
        
        # Check alpha
        def fast_alpha():
            remaining = list(range(n_target))
            count = 0
            while remaining:
                best_idx = 0
                best_deg = float('inf')
                for i, v in enumerate(remaining):
                    deg = sum(1 for u in adj[v] if u in remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_idx = i
                        if deg == 0:
                            break
                count += 1
                v = remaining.pop(best_idx)
                remaining = [u for u in remaining if u not in adj[v]]
            return count
        
        alpha = fast_alpha()
        if alpha <= 10:
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n_target, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try to construct a 39-vertex graph
    result = _construct_polynomial(39, deadline)
    if result is not None:
        return result
    
    # Try 38
    result = _construct_polynomial(38, deadline)
    if result is not None:
        return result
    
    # Try 37
    result = _construct_polynomial(37, deadline)
    if result is not None:
        return result
    
    # Fallback: return the seed's 30-vertex construction
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
# EVOLVE-BLOCK-END
