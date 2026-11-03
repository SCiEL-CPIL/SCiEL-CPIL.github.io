import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
    1. Try LUW construction with many rule variants (fast, deterministic).
    2. Try hash-consing with improved boundary matching.
    3. Verify girth exactly and return the best valid graph found.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_n = float('inf')
    best_edges = None

    if k == 3 and g == 16:
        # --- Strategy 1: LUW with many rule variants ---
        for d in [7, 8, 9, 10, 11]:
            if time.time() > deadline - 20:
                break
            for rule in _generate_rule_variants(k, d):
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if n_luw <= 936:
                                return n_luw, edges_luw
                except Exception:
                    continue

        # --- Strategy 2: Hash-consing with improved boundary matching ---
        if time.time() < deadline - 10:
            try:
                n_hc, edges_hc = _construct_hash_consolidated(k, g)
                if n_hc > 0 and _is_valid_3reg(n_hc, edges_hc, k):
                    if _has_girth_at_least_fast(n_hc, edges_hc, g):
                        if n_hc < best_n:
                            best_n = n_hc
                            best_edges = edges_hc
            except Exception:
                pass

        if best_n != float('inf'):
            return best_n, best_edges

    # --- Generic fallback for other (k, g) ---
    min_d = max(3, g // 2 - 2)
    max_d = min(g + 5, 12)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in _generate_rule_variants(k, d):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue

    # Last resort
    n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, _generate_rule_variants(k, max_d + 3)[0])
    return n_luw, edges_luw


def _construct_hash_consolidated(k, g):
    """
    Hash-consing construction: build k-regular tree to depth radius,
    then match boundary vertices using a structure-aware pairing that
    maximizes tree-distance between matched vertices.
    """
    radius = (g - 1) // 2  # For g=16, radius=7
    q = k

    # Build k-regular tree of depth radius
    vertices = {}
    edges = []
    vertex_list = []

    def get_vertex(path):
        if path not in vertices:
            vertices[path] = len(vertex_list)
            vertex_list.append(path)
        return vertices[path]

    root_path = ()
    get_vertex(root_path)

    queue = deque([root_path])
    while queue:
        path = queue.popleft()
        depth = len(path)
        if depth >= radius:
            continue
        if depth == 0:
            available = list(range(k))
        else:
            last_dir = path[-1]
            available = [d for d in range(k) if d != last_dir]

        for d in available:
            new_path = path + (d,)
            idx = get_vertex(new_path)
            parent_idx = vertices[path]
            edges.append((parent_idx, idx))
            queue.append(new_path)

    n_tree = len(vertex_list)

    boundary_paths = [p for p in vertex_list if len(p) == radius]
    n_boundary = len(boundary_paths)

    if n_boundary == 0:
        return n_tree, edges

    if k == 3:
        # For k=3, radius=7: boundary has 3*2^6 = 192 vertices.
        # Each boundary vertex needs 2 more edges.
        # 
        # Key insight: two boundary vertices connected by an edge create a cycle
        # of length 2*(radius - lcp_length) + 1 where lcp_length is their longest
        # common prefix length. For girth >= 16, we need 2*(7 - lcp) + 1 >= 16,
        # i.e., lcp <= 0. So connected vertices must have NO common prefix!
        #
        # This means: vertex in group 0 (d1=0) can only connect to group 1 or 2.
        # Within cross-group pairs, ANY pairing works since lcp=0 always.
        # But we need the matching to be a perfect matching (each vertex gets exactly 2 edges).
        #
        # Strategy: 
        # - Group 0 (64 vertices) -> connect to Group 1 (64) and Group 2 (64)
        # - Group 1 (64 vertices) -> connect to Group 0 (64) and Group 2 (64)  
        # - Group 2 (64 vertices) -> connect to Group 0 (64) and Group 1 (64)
        # Each group provides 64 edges to each of the other two groups.
        # Total: 3 * 64 * 2 / 2 = 192 edges (each boundary vertex gets 2 edges).
        #
        # Use a simple bijection: vertex i in group A connects to vertex i in group B.
        # Since all cross-group pairs have lcp=0, this is always valid for girth.
        
        groups = {0: [], 1: [], 2: []}
        for i, p in enumerate(boundary_paths):
            groups[p[0]].append((i, p))
        
        # Sort each group by path for determinism
        for g_idx in range(3):
            groups[g_idx].sort(key=lambda x: x[1])
        
        added_edges = set()
        n_per_group = len(groups[0])  # Should be 64
        
        def add_edge(path_a, path_b):
            ga = vertices[path_a]
            gb = vertices[path_b]
            if ga == gb:
                return
            edge = (min(ga, gb), max(ga, gb))
            if edge not in added_edges:
                added_edges.add(edge)
                edges.append(edge)
        
        # Group 0 <-> Group 1: vertex i in G0 connects to vertex i in G1
        for i in range(n_per_group):
            add_edge(groups[0][i][1], groups[1][i][1])
        
        # Group 0 <-> Group 2: vertex i in G0 connects to vertex i in G2
        for i in range(n_per_group):
            add_edge(groups[0][i][1], groups[2][i][1])
        
        # Group 1 <-> Group 2: vertex i in G1 connects to vertex i in G2
        for i in range(n_per_group):
            add_edge(groups[1][i][1], groups[2][i][1])
        
    return n_tree, edges


def _is_valid_3reg(n, edges, k):
    """Check that the graph is k-regular and simple."""
    if len(edges) != n * k // 2:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u in adj[v]:
            return False
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    return True


def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks from every vertex.
    Returns True if no cycle shorter than g exists.
    Uses BFS with early termination for efficiency.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    half_g = g // 2
    
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle through u-v
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    More variants with different patterns to find smaller valid graphs.
    """
    variants = []
    seen = set()
    
    def make_rule(base_rules, pattern_type, seed_shift=0):
        """Create a rule dict from base rules and pattern type."""
        rule = {}
        for c in range(2, max_d + 1):
            if c in base_rules:
                rule[c] = base_rules[c]
            else:
                if pattern_type == 0:
                    r, c0 = (c - 5 + seed_shift) % 4, c - (c - 5 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 1:
                    r, c0 = (c - 4 + seed_shift) % 4, c - (c - 4 + seed_shift) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 2:
                    r, c0 = (c - 5 + seed_shift) % 3, c - (c - 5 + seed_shift) % 3
                    rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern_type == 3:
                    r, c0 = (c - 5 + seed_shift) % 5, c - (c - 5 + seed_shift) % 5
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1)][r % 5]
                elif pattern_type == 4:
                    r, c0 = (c - 6 + seed_shift) % 6, c - (c - 6 + seed_shift) % 6
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 1, 1), (c0 + 2, 1)][r % 6]
                elif pattern_type == 5:
                    # Fibonacci-like
                    rule[c] = (1, c - 1) if c % 2 == 0 else (1, c - 2)
                elif pattern_type == 6:
                    # Linear: (1, c-1) always
                    rule[c] = (1, c - 1)
                elif pattern_type == 7:
                    # Alternating
                    rule[c] = (1, c - 1) if c % 2 == 0 else (2, c - 2)
                elif pattern_type == 8:
                    # Periodic with period 3
                    r = (c + seed_shift) % 3
                    rule[c] = [(1, 1), (1, 2), (2, 1)][r]
                elif pattern_type == 9:
                    # Periodic with period 4
                    r = (c + seed_shift) % 4
                    rule[c] = [(1, 1), (1, 2), (2, 1), (1, 3)][r]
                else:
                    rule[c] = (1, 1)
        return rule
    
    # Generate many variants
    base_rule_sets = [
        {2: (1, 1), 3: (2, 1), 4: (1, 2)},
        {2: (1, 1), 3: (1, 2), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 2), 4: (1, 1)},
        {2: (1, 1), 3: (2, 1), 4: (2, 1)},
        {2: (1, 1), 3: (1, 1), 4: (1, 2)},
        {2: (1, 1), 3: (1, 1), 4: (1, 1)},
        {2: (2, 1), 3: (1, 1), 4: (1, 1)},
    ]
    
    for base in base_rule_sets:
        for ptype in range(10):
            for shift in range(4):
                rule = make_rule(base, ptype, shift)
                key = tuple(sorted(rule.items()))
                if key not in seen:
                    seen.add(key)
                    variants.append(rule)
    
    return variants


def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with given degree k, depth d, and rule.
    """
    q = k if k >= 3 else 3

    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
