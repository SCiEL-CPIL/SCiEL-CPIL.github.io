"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _build_adj(edges, n=18):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_k4_free(adj, n=18):
    """Check K4-free by checking every triangle has no common neighbor."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        # u,v,w is a triangle; check for K4
                        if adj[u] & adj[v] & adj[w]:
                            return False
    return True

def _get_triangles(adj, n=18):
    """Get all triangles as frozensets."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        triangles.append(frozenset((u, v, w)))
    return triangles

def _max_triangle_free_induced(adj, n=18, triangles=None):
    """
    Find the largest triangle-free induced subgraph.
    This is equivalent to: find the maximum subset of vertices containing no triangle.
    Equivalently: minimum vertex cover of the triangle hypergraph.
    We solve via branch and bound on the triangle hypergraph.
    """
    if triangles is None:
        triangles = _get_triangles(adj, n)
    
    if not triangles:
        return n
    
    # We want to find the minimum number of vertices to remove so that no triangle remains.
    # This is the minimum vertex cover of a 3-uniform hypergraph.
    # Branch and bound: pick a triangle, branch on removing one of its 3 vertices.
    
    best = [n]  # best cover size found (we want to minimize)
    all_vertices = set(range(n))
    
    def branch(remaining_triangles, covered, depth):
        """
        remaining_triangles: list of frozensets of triangles not yet covered
        covered: set of vertices already in the cover
        """
        if not remaining_triangles:
            if len(covered) < best[0]:
                best[0] = len(covered)
            return
        
        # Lower bound: we need at least ceil(len(remaining_triangles) / max_occurrence) more vertices
        # Simpler bound: at least 1 more if there are triangles
        if len(covered) + 1 >= best[0]:
            return
        
        # Pick the triangle with the fewest uncovered vertices (most constrained)
        tri = None
        min_uncovered = 4
        for t in remaining_triangles:
            unc = len(t - covered)
            if unc < min_uncovered:
                min_uncovered = unc
                tri = t
            if unc <= 1:
                break
        
        if tri is None:
            return
        
        # Branch: for each vertex in tri not yet covered, add it to cover
        candidates = [v for v in tri if v not in covered]
        if not candidates:
            # Triangle already covered
            return
        
        for v in candidates:
            new_covered = covered | {v}
            # Filter remaining triangles
            new_remaining = [t for t in remaining_triangles if v not in t]
            branch(new_remaining, new_covered, depth + 1)
    
    branch(triangles, set(), 0)
    return n - best[0]

def _max_triangle_free_induced_v2(adj, n=18, triangles=None):
    """
    Improved: Use the complement approach.
    A set S is triangle-free iff it contains no triangle from the graph.
    We want max |S| such that S contains no triangle.
    
    This is the same as: max independent set in the 3-uniform hypergraph's "triangle graph"
    where two vertices are adjacent if they appear in a common triangle... no that's not right.
    
    Actually: S is triangle-free iff for every triangle {a,b,c}, S does NOT contain all of a,b,c.
    This is a hitting set / vertex cover problem on the triangle hypergraph.
    Max triangle-free = n - min vertex cover of triangle hypergraph.
    
    Let me use a better branch and bound with memoization and stronger pruning.
    """
    if triangles is None:
        triangles = _get_triangles(adj, n)
    
    if not triangles:
        return n
    
    # Convert to list of sets for faster operations
    tri_list = [set(t) for t in triangles]
    
    # Build vertex-to-triangles index
    vert_tris = {v: [] for v in range(n)}
    for i, t in enumerate(tri_list):
        for v in t:
            vert_tris[v].append(i)
    
    # We'll do branch and bound to find minimum vertex cover
    # State: which triangles are still uncovered
    # Use bitmasks for triangles if count is manageable
    
    num_tris = len(tri_list)
    
    # Precompute for each vertex: bitmask of triangles it covers
    vert_mask = [0] * n
    for i, t in enumerate(tri_list):
        for v in t:
            vert_mask[v] |= (1 << i)
    
    best_cover = [n + 1]
    
    def bnb(uncovered_mask, cover_size, available_mask):
        """
        uncovered_mask: bitmask of triangles not yet covered
        cover_size: number of vertices in current cover
        available_mask: bitmask of vertices not yet used (or -1 for all)
        """
        if uncovered_mask == 0:
            if cover_size < best_cover[0]:
                best_cover[0] = cover_size
            return
        
        if cover_size >= best_cover[0]:
            return
        
        # Lower bound: each remaining vertex can cover at most max_cover triangles
        # Simple bound: need at least 1 more
        if cover_size + 1 >= best_cover[0]:
            return
        
        # Pick the most constrained uncovered triangle
        # Find an uncovered triangle
        # Get the first set bit
        lb = uncovered_mask & (-uncovered_mask)
        tri_idx = lb.bit_length() - 1
        tri = tri_list[tri_idx]
        
        # Branch on vertices of this triangle
        for v in tri:
            new_uncovered = uncovered_mask & ~vert_mask[v]
            bnb(new_uncovered, cover_size + 1, available_mask & ~(1 << v))
    
    all_mask = (1 << num_tris) - 1
    bnb(all_mask, 0, (1 << n) - 1)
    
    return n - best_cover[0]

def _score_graph(adj, n=18):
    """Compute the score: largest triangle-free induced subgraph size."""
    triangles = _get_triangles(adj, n)
    return _max_triangle_free_induced_v2(adj, n, triangles)

def _random_k4_free_graph(n=18, seed=None):
    """Generate a random K4-free graph."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        # Check if adding (u,v) creates a K4
        common = adj[u] & adj[v]
        common_list = list(common)
        ok = True
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                w, x = common_list[i], common_list[j]
                if x in adj[w]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
    
    return adj, list(edge_set)

def _add_edge_safe(adj, u, v, n=18):
    """Try to add edge (u,v). Returns True if added without creating K4."""
    if v in adj[u]:
        return True
    common = adj[u] & adj[v]
    common_list = list(common)
    for i in range(len(common_list)):
        for j in range(i+1, len(common_list)):
            w, x = common_list[i], common_list[j]
            if x in adj[w]:
                return False
    adj[u].add(v)
    adj[v].add(u)
    return True

def _remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def _local_search(adj, n=18, max_time=100, seed=None):
    """Local search: try edge additions and removals to minimize score."""
    rng = random.Random(seed)
    start = time.time()
    
    triangles = _get_triangles(adj, n)
    current_score = _max_triangle_free_induced_v2(adj, n, triangles)
    best_score = current_score
    best_adj = [set(s) for s in adj]
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    iteration = 0
    while time.time() - start < max_time:
        iteration += 1
        
        # Choose a random operation
        if rng.random() < 0.5 and len([1 for u in range(n) for v in adj[u] if v > u]) > 0:
            # Remove a random edge
            existing = [(u, v) for u in range(n) for v in adj[u] if v > u]
            if not existing:
                break
            u, v = rng.choice(existing)
            _remove_edge(adj, u, v)
            
            new_triangles = _get_triangles(adj, n)
            new_score = _max_triangle_free_induced_v2(adj, n, new_triangles)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [set(s) for s in adj]
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
        else:
            # Add a random edge
            u, v = rng.choice(all_edges)
            if v in adj[u]:
                continue
            if _add_edge_safe(adj, u, v, n):
                new_triangles = _get_triangles(adj, n)
                new_score = _max_triangle_free_induced_v2(adj, n, new_triangles)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [set(s) for s in adj]
                else:
                    _remove_edge(adj, u, v)
    
    return best_adj, best_score

def _targeted_search(adj, n=18, max_time=100, seed=None):
    """
    More targeted search: focus on the largest triangle-free subgraph and
    try to add edges within it to break triangles.
    """
    rng = random.Random(seed)
    start = time.time()
    
    current_score = _score_graph(adj, n)
    best_score = current_score
    best_adj = [set(s) for s in adj]
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() - start < max_time:
        triangles = _get_triangles(adj, n)
        if not triangles:
            break
        
        # Find a large triangle-free set by taking all vertices and removing
        # Actually, let's just try random edge additions
        # Focus: add edges that create triangles (to reduce triangle-free sets)
        
        # Try adding edges between vertices that are in many triangles together
        candidates = []
        for u, v in all_edges:
            if v not in adj[u]:
                common = adj[u] & adj[v]
                if common:
                    # Adding this edge creates |common| new triangles
                    candidates.append((len(common), u, v))
        
        if not candidates:
            break
        
        # Sort by number of triangles created (descending) and pick from top
        candidates.sort(reverse=True)
        top = candidates[:min(20, len(candidates))]
        
        # Randomly pick from top candidates
        _, u, v = rng.choice(top)
        
        if _add_edge_safe(adj, u, v, n):
            new_score = _score_graph(adj, n)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [set(s) for s in adj]
            else:
                _remove_edge(adj, u, v)
        # Occasionally try removing an edge
        elif rng.random() < 0.2:
            existing = [(u, v) for u in range(n) for v in adj[u] if v > u]
            if existing:
                u, v = rng.choice(existing)
                _remove_edge(adj, u, v)
                new_score = _score_graph(adj, n)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [set(s) for s in adj]
                else:
                    adj[u].add(v)
                    adj[v].add(u)
    
    return best_adj, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n = 18
    
    # Start with complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    base_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    base_edges.append([a, b])
    
    base_adj = _build_adj(base_edges, n)
    base_score = _score_graph(base_adj, n)
    
    best_adj = [set(s) for s in base_adj]
    best_score = base_score
    
    # Try multiple random constructions
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Generate random K4-free graph
        adj, edges = _random_k4_free_graph(n, seed=trial * 1000 + 42)
        
        if not _is_k4_free(adj, n):
            continue
        
        score = _score_graph(adj, n)
        
        if score < best_score:
            best_score = score
            best_adj = [set(s) for s in adj]
        
        # Local search on this graph
        time_left = deadline - time.time()
        if time_left > 5:
            ls_adj, ls_score = _local_search([set(s) for s in adj], n, max_time=min(20, time_left - 2), seed=trial)
            if ls_score < best_score:
                best_score = ls_score
                best_adj = [set(s) for s in ls_adj]
        
        # Also try targeted search
        time_left = deadline - time.time()
        if time_left > 5:
            ts_adj, ts_score = _targeted_search([set(s) for s in adj], n, max_time=min(20, time_left - 2), seed=trial + 1000)
            if ts_score < best_score:
                best_score = ts_score
                best_adj = [set(s) for s in ts_adj]
    
    # Also do local search on the base graph
    time_left = deadline - time.time()
    if time_left > 5:
        ls_adj, ls_score = _local_search([set(s) for s in base_adj], n, max_time=min(30, time_left - 2), seed=9999)
        if ls_score < best_score:
            best_score = ls_score
            best_adj = [set(s) for s in ls_adj]
    
    # Final targeted search on best
    time_left = deadline - time.time()
    if time_left > 5:
        ts_adj, ts_score = _targeted_search([set(s) for s in best_adj], n, max_time=min(30, time_left - 2), seed=7777)
        if ts_score < best_score:
            best_score = ts_score
            best_adj = [set(s) for s in ts_adj]
    
    # Extract edges from best_adj
    result_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if v > u:
                result_edges.append([u, v])
    
    return result_edges
# EVOLVE-BLOCK-END
