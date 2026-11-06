"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_c4_bitmask(adj, n):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                return True
    return False

def _max_independent_set_size_bitmask(adj, n):
    """Find maximum independent set size using bitmask branch and bound."""
    # Greedy lower bound
    remaining = (1 << n) - 1
    alpha_lb = 0
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = -1
        best_deg = n
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            r ^= lsb
        if best_v == -1:
            break
        alpha_lb += 1
        remaining &= ~adj[best_v]
        remaining &= ~(1 << best_v)
    
    # Branch and bound for upper bound
    best = [0]
    def branch(remaining, current_size):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + remaining.bit_count() <= best[0]:
            return
        # Pick a vertex
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        # Include v
        branch(remaining & ~adj[v] & ~(1 << v), current_size + 1)
        # Exclude v
        branch(remaining & ~(1 << v), current_size)
    
    branch((1 << n) - 1, 0)
    return best[0]

def _verify_graph(n, edges):
    """Verify graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        if u >= n or v >= n or u == v:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check C4
    if _has_c4_bitmask(adj, n):
        return False
    # Check independence number < 11
    alpha = _max_independent_set_size_bitmask(adj, n)
    if alpha >= 11:
        return False
    return True

def _build_polarity_graph_q5():
    """
    Construct polarity graph from PG(2,5) - a known C4-free graph.
    PG(2,q) has (q^2+q+1) points. For q=5, that's 31 points.
    Two points are adjacent if they are perpendicular under the polarity.
    """
    q = 5
    # Points of PG(2,q) as equivalence classes of non-zero vectors in F_q^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(x * inv) % q for x in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)  # Should be q^2 + q + 1 = 31
    
    # Polarity: x·y = x0*y0 + x1*y1 + x2*y2 (symmetric bilinear form)
    # Two points are adjacent if their dot product is 0
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0] * points[j][0] + points[i][1] * points[j][1] + points[i][2] * points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
    
    return n, edges

def _try_add_edge(adj, n, u, v):
    """Try to add edge (u,v) maintaining C4-free property."""
    if u == v or u >= n or v >= n:
        return False
    if adj[u] & (1 << v):
        return False
    # Check C4: for any common neighbor w of u and v, adding edge creates triangle
    # but we need to check that no two vertices have two common neighbors after adding
    # Adding edge (u,v): for any x != u,v, if x is neighbor of both u and v,
    # then u and v now share neighbor x. Need to ensure u and v don't already share a neighbor.
    common = adj[u] & adj[v]
    if common:
        return False
    # Also need to check: for any pair (a,b) where a is neighbor of u, b is neighbor of v,
    # a != b, a != v, b != u: after adding edge, do a and b gain a common neighbor?
    # Actually, the C4 condition is: no two vertices have two common neighbors.
    # Adding edge (u,v) affects common neighbors of:
    # - u and v: they now share each other as neighbors (but that's fine, just one)
    # - For any x neighbor of u: x and v now both neighbor u. Check if x and v already share another neighbor.
    # - For any y neighbor of v: u and y now both neighbor v. Check if u and y already share another neighbor.
    
    # Check: for each x in N(u), x and v must not have two common neighbors after adding
    # Common neighbors of x and v after adding: N(x) ∩ N(v) ∪ {u} (if u in N(x) and u in N(v))
    # u is in N(x) always (since x in N(u)), and u will be in N(v) after adding.
    # So u becomes a common neighbor of x and v.
    # Need: |N(x) ∩ N(v)| <= 0 (i.e., x and v have no common neighbors before adding)
    # Wait, that's not right. Let me think again.
    
    # After adding (u,v):
    # N'(u) = N(u) ∪ {v}
    # N'(v) = N(v) ∪ {u}
    
    # For any pair (a,b), count common neighbors in N'.
    # The only pairs whose common neighbor count can change are those involving u or v.
    
    # Pair (u,v): common neighbors = N(u) ∩ N(v). Must be 0 (already checked).
    # Pair (u,x) for x in N(v), x != u: common neighbors = N(u) ∩ N(x) ∪ {v} (if v in N(u) and v in N(x))
    # v is in N'(u). v is in N'(x) iff v in N(x) (i.e., x in N(v)).
    # So if x in N(v), then v is a common neighbor of u and x in N'.
    # Also, any y in N(u) ∩ N(x) is a common neighbor.
    # Total common neighbors of u and x in N' = |N(u) ∩ N(x)| + (1 if x in N(v) else 0)
    # Must be <= 1.
    # Since graph was C4-free before, |N(u) ∩ N(x)| <= 1.
    # If x in N(v), then we need |N(u) ∩ N(x)| <= 0, i.e., u and x have no common neighbor.
    
    # Similarly for pair (v,x) for x in N(u), x != v.
    
    for x in range(n):
        if x == u or x == v:
            continue
        # Check pair (u, x)
        if adj[v] & (1 << x):  # x is neighbor of v
            # u and x must have no common neighbor
            if adj[u] & adj[x]:
                return False
        # Check pair (v, x)
        if adj[u] & (1 << x):  # x is neighbor of u
            # v and x must have no common neighbor
            if adj[v] & adj[x]:
                return False
    
    return True

def _construct_large_graph(deadline):
    """
    Try to construct a large C4-free graph with alpha < 11.
    Strategy: Start with polarity graph PG(2,5) (31 vertices), try to add vertices/edges.
    """
    start_time = time.time()
    
    # Start with polarity graph
    n0, edges0 = _build_polarity_graph_q5()
    
    adj = [0] * n0
    for u, v in edges0:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    best_n = n0
    best_edges = edges0[:]
    
    # Verify initial graph
    if time.time() < deadline:
        if _verify_graph(n0, edges0):
            best_n = n0
            best_edges = edges0[:]
    
    # Try to add vertices one by one
    # For each new vertex, try all possible neighborhoods that maintain C4-free
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        current_n = len(adj)
        new_v = current_n
        adj.append(0)
        
        # Try to find a valid neighborhood for new_v
        # The neighborhood must be such that:
        # 1. No C4 is created
        # 2. The independence number stays < 11
        
        # Greedy: try to connect to as many vertices as possible
        # while maintaining C4-free
        
        # For C4-free: if new_v connects to u and v, then u and v must have no common neighbor
        # (other than new_v itself, but new_v is only connected to them)
        # Actually, for any two neighbors u,v of new_v, u and v must not share any common neighbor
        # (because new_v would be a common neighbor, and if they share another, that's C4)
        
        # So the neighborhood of new_v must be an independent set in the "common neighbor" sense:
        # For any u,v in N(new_v), N(u) ∩ N(v) = empty
        
        # Let's try a greedy approach
        candidates = list(range(current_n))
        random.shuffle(candidates)
        
        neighborhood = []
        for u in candidates:
            if time.time() > deadline:
                break
            # Check if we can add u to neighborhood
            ok = True
            for v in neighborhood:
                if adj[u] & adj[v]:
                    ok = False
                    break
            if ok:
                neighborhood.append(u)
        
        # Set up the new vertex
        for u in neighborhood:
            adj[new_v] |= (1 << u)
            adj[u] |= (1 << new_v)
        
        # Check if graph is still C4-free
        if not _has_c4_bitmask(adj, current_n + 1):
            # Check independence number
            if time.time() < deadline:
                alpha = _max_independent_set_size_bitmask(adj, current_n + 1)
                if alpha < 11:
                    best_n = current_n + 1
                    best_edges = []
                    for i in range(current_n + 1):
                        for j in range(i + 1, current_n + 1):
                            if adj[i] & (1 << j):
                                best_edges.append([i, j])
        
        # If we couldn't add a vertex, try different randomizations
        adj = adj[:current_n]
    
    return best_n, best_edges

def _construct_by_search(deadline):
    """
    Alternative: try to build a graph from scratch using greedy + local search.
    """
    start_time = time.time()
    
    # Try different sizes
    for target_n in range(39, 29, -1):
        if time.time() > deadline:
            break
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random initial graph
            adj = [0] * target_n
            edges = []
            
            # Start with random edges, maintain C4-free
            random.seed(trial)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        if _try_add_edge(adj, target_n, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            edges.append([i, j])
            
            # Check if C4-free
            if _has_c4_bitmask(adj, target_n):
                continue
            
            # Check independence number
            if time.time() > deadline:
                break
            
            alpha = _max_independent_set_size_bitmask(adj, target_n)
            if alpha < 11:
                # Found a valid graph
                return target_n, edges
            
            # Try to improve: add edges to reduce independence number
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if not (adj[i] & (1 << j)):
                            if _try_add_edge(adj, target_n, i, j):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                edges.append([i, j])
                                improved = True
                                # Check C4-free
                                if _has_c4_bitmask(adj, target_n):
                                    # Roll back
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    edges.pop()
                                    improved = False
                                    break
                                # Check independence
                                if time.time() < deadline:
                                    alpha = _max_independent_set_size_bitmask(adj, target_n)
                                    if alpha < 11:
                                        return target_n, edges[:]
                    if improved:
                        break
            
            if not improved and time.time() < deadline:
                alpha = _max_independent_set_size_bitmask(adj, target_n)
                if alpha < 11:
                    return target_n, edges[:]
    
    return 0, []

def construct():
    """Construct a C4-free graph with no independent set of size 11."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, try the polarity graph approach
    n, edges = _construct_large_graph(deadline)
    
    if n >= 31 and _verify_graph(n, edges):
        return {'n': n, 'edges': edges}
    
    # Try search-based approach
    n2, edges2 = _construct_by_search(deadline)
    
    if n2 > n and _verify_graph(n2, edges2):
        return {'n': n2, 'edges': edges2}
    
    # Fallback: ten disjoint triangles
    if n >= 30:
        return {'n': n, 'edges': edges}
    
    edges_fallback = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges_fallback += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges_fallback}

# EVOLVE-BLOCK-END
