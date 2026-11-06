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

def _has_c4_fast(adj, n):
    """Faster C4 check: for each edge (u,v), check if any pair of common neighbors exists."""
    for u in range(n):
        au = adj[u]
        for v_bit in au:
            v = v_bit.bit_length() - 1
            if v > u:
                common = au & adj[v]
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
    best = [alpha_lb]
    def branch(remaining, current_size):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + remaining.bit_count() <= best[0]:
            return
        # Pick a vertex with max degree in remaining for better pruning
        best_v = -1
        best_deg = -1
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            r ^= lsb
        
        # Include v
        branch(remaining & ~adj[best_v] & ~(1 << best_v), current_size + 1)
        # Exclude v
        branch(remaining & ~(1 << best_v), current_size)
    
    branch((1 << n) - 1, 0)
    return best[0]

def _independent_set_exists(adj, n, k):
    """Check if there exists an independent set of size k. Returns True if exists."""
    # Use branch and bound to find if independent set of size k exists
    found = [False]
    
    def branch(remaining, current_size):
        if found[0]:
            return
        if current_size >= k:
            found[0] = True
            return
        if not remaining:
            return
        if current_size + remaining.bit_count() < k:
            return
        
        # Pick vertex with max degree in remaining
        best_v = -1
        best_deg = -1
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            r ^= lsb
        
        # Include v first (more likely to find large independent set)
        branch(remaining & ~adj[best_v] & ~(1 << best_v), current_size + 1)
        if not found[0]:
            branch(remaining & ~(1 << best_v), current_size)
    
    branch((1 << n) - 1, 0)
    return found[0]

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

def _build_projective_plane_incidence(q):
    """
    Build the incidence graph of PG(2,q) as a bipartite graph.
    Returns (n_points, n_lines, point_to_lines, line_to_points)
    where point i is incident to line j if point i lies on line j.
    """
    # Points of PG(2,q)
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
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
    
    # Lines of PG(2,q) - dual of points
    # A line is an equivalence class of non-zero linear forms
    lines = []
    seen_lines = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(x * inv) % q for x in vec]
                        break
                key = tuple(vec)
                if key not in seen_lines:
                    seen_lines.add(key)
                    lines.append(key)
    
    n_pts = len(points)
    n_lns = len(lines)
    
    # Incidence: point p is on line L if dot(p, L) == 0
    point_to_lines = [set() for _ in range(n_pts)]
    line_to_points = [set() for _ in range(n_lns)]
    
    for i, p in enumerate(points):
        for j, l in enumerate(lines):
            dot = (p[0] * l[0] + p[1] * l[1] + p[2] * l[2]) % q
            if dot == 0:
                point_to_lines[i].add(j)
                line_to_points[j].add(i)
    
    return n_pts, n_lns, point_to_lines, line_to_points

def _try_add_edge(adj, n, u, v):
    """Try to add edge (u,v) maintaining C4-free property."""
    if u == v or u >= n or v >= n:
        return False
    if adj[u] & (1 << v):
        return False
    common = adj[u] & adj[v]
    if common:
        return False
    
    for x in range(n):
        if x == u or x == v:
            continue
        if adj[v] & (1 << x):
            if adj[u] & adj[x]:
                return False
        if adj[u] & (1 << x):
            if adj[v] & adj[x]:
                return False
    
    return True

def _construct_large_graph(deadline):
    """
    Try to construct a large C4-free graph with alpha < 11.
    Strategy: Start with polarity graph PG(2,5) (31 vertices), try to add vertices/edges.
    Also try projective plane incidence contraction for larger graphs.
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
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        current_n = len(adj)
        new_v = current_n
        adj.append(0)
        
        # Try different random orderings
        candidates = list(range(current_n))
        random.shuffle(candidates)
        
        neighborhood = []
        for u in candidates:
            if time.time() > deadline:
                break
            ok = True
            for v in neighborhood:
                if adj[u] & adj[v]:
                    ok = False
                    break
            if ok:
                neighborhood.append(u)
        
        for u in neighborhood:
            adj[new_v] |= (1 << u)
            adj[u] |= (1 << new_v)
        
        if not _has_c4_bitmask(adj, current_n + 1):
            if time.time() < deadline:
                alpha = _max_independent_set_size_bitmask(adj, current_n + 1)
                if alpha < 11:
                    best_n = current_n + 1
                    best_edges = []
                    for i in range(current_n + 1):
                        for j in range(i + 1, current_n + 1):
                            if adj[i] & (1 << j):
                                best_edges.append([i, j])
        
        adj = adj[:current_n]
    
    return best_n, best_edges

def _construct_by_search(deadline):
    """
    Alternative: try to build a graph from scratch using greedy + local search.
    """
    start_time = time.time()
    
    for target_n in range(39, 29, -1):
        if time.time() > deadline:
            break
        
        for trial in range(30):
            if time.time() > deadline:
                break
            
            adj = [0] * target_n
            edges = []
            
            random.seed(trial * 1000 + target_n)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        if _try_add_edge(adj, target_n, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            edges.append([i, j])
            
            if _has_c4_bitmask(adj, target_n):
                continue
            
            if time.time() > deadline:
                break
            
            alpha = _max_independent_set_size_bitmask(adj, target_n)
            if alpha < 11:
                return target_n, edges
            
            # Try to improve: add edges to reduce independence number
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(target_n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, target_n):
                        if not (adj[i] & (1 << j)):
                            if _try_add_edge(adj, target_n, i, j):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                edges.append([i, j])
                                improved = True
                                if _has_c4_bitmask(adj, target_n):
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    edges.pop()
                                    improved = False
                                    break
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
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
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
