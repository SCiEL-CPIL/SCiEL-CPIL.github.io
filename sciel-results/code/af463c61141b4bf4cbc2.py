# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigvalsh as _eigvalsh

def _spectral_radius_of_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    eigvals = _eigvalsh(S)
    return np.max(np.abs(eigvals))

def _harmonic_init(graph, seed=42):
    """Generate initial signs by solving for harmonic 1-cochain to minimize energy.
    
    The signed adjacency matrix S has entries S[u,v] = sign(e) for edge e=(u,v).
    We want to minimize ||S||_spec. A good heuristic: find signs that make S
    as "anti-symmetric" as possible in a weighted sense, which corresponds to
    minimizing the quadratic form x^T S x over unit x, or equivalently making
    the eigenvalues as close to zero as possible.
    
    Approach: Use the fact that for a signed graph, the spectrum relates to
    the cycle structure. We can initialize by:
    1. Start with all +1
    2. For each fundamental cycle, if the product is +1, flip one edge to make it -1
    3. This creates a "frustrated" system that tends to have smaller spectral radius
    
    Better: Use gauge fixing. Pick a spanning tree, set tree edges to +1,
    then optimize the non-tree edges. The spectral radius depends only on
    the products around cycles (gauge invariance).
    """
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Build adjacency for spanning tree
    adj = {i: [] for i in range(n)}
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # BFS to find spanning tree
    visited = [False] * n
    tree_edges = set()
    non_tree_edges = []
    queue = [0]
    visited[0] = True
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
            elif ei not in tree_edges and (v, u) not in [(a, b) for a, b in edges]:
                non_tree_edges.append(ei)
    
    # Actually, let's just identify non-tree edges more carefully
    # An edge is non-tree if it's not in the spanning tree
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    
    # Initialize: tree edges = +1, non-tree edges to be optimized
    signs = np.ones(num_edges, dtype=np.int8)
    
    # For each non-tree edge, determine its sign to create frustration
    # The fundamental cycle for non-tree edge e=(u,v) is: path from u to v in tree + edge e
    # We want the product around each fundamental cycle to be -1 for optimal frustration
    
    # First, compute parent pointers from BFS tree
    parent = [-1] * n
    parent_edge = [-1] * n
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                parent_edge[v] = ei
                queue.append(v)
    
    def path_product(u, v):
        """Compute product of signs on path from u to v in tree."""
        # Find LCA
        depth = [0] * n
        stack = [0]
        while stack:
            node = stack.pop()
            for child in range(n):
                if parent[child] == node:
                    depth[child] = depth[node] + 1
                    stack.append(child)
        
        # Path from u to LCA and from v to LCA
        prod = 1
        # Go up from u to LCA
        au, av = u, v
        while au != av:
            if depth[au] > depth[av]:
                prod *= signs[parent_edge[au]]
                au = parent[au]
            elif depth[av] > depth[au]:
                prod *= signs[parent_edge[av]]
                av = parent[av]
            else:
                prod *= signs[parent_edge[au]]
                prod *= signs[parent_edge[av]]
                au = parent[au]
                av = parent[av]
        return prod
    
    # For each non-tree edge, set its sign so that the fundamental cycle product is -1
    for ei in non_tree_edges:
        u, v = edges[ei]
        cycle_prod = path_product(u, v)
        # We want cycle_prod * signs[ei] = -1
        # So signs[ei] = -1 / cycle_prod = -cycle_prod (since signs are ±1)
        signs[ei] = -cycle_prod
    
    return signs

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Precompute edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Precompute cycle edge indices for fast connectivity
    # Top cycle edges
    top_cycle_edges = []
    for i in range(m):
        u, v = i, (i + 1) % m
        top_cycle_edges.append(edge_idx[(u, v)])
    
    # Bottom cycle edges
    bot_cycle_edges = []
    for i in range(m):
        u, v = m + i, m + (i + 1) % m
        bot_cycle_edges.append(edge_idx[(u, v)])
    
    # 4-cycle edges
    four_cycle_edges = []
    for i in range(m):
        u1, u2 = i, (i + 1) % m
        u3, u4 = m + (i + 1) % m, m + i
        ec = []
        for (a, b) in [(u1, u2), (u2, u3), (u3, u4), (u4, u1)]:
            if (a, b) in edge_idx:
                ec.append(edge_idx[(a, b)])
            else:
                ec.append(edge_idx[(b, a)])
        four_cycle_edges.append(ec)
    
    def fast_is_connected(signs):
        """O(m) connectivity check using precomputed cycle edges."""
        prod_top = 1
        for ei in top_cycle_edges:
            prod_top *= signs[ei]
        if prod_top == -1:
            return True
        
        prod_bot = 1
        for ei in bot_cycle_edges:
            prod_bot *= signs[ei]
        if prod_bot == -1:
            return True
        
        for ec in four_cycle_edges:
            prod4 = 1
            for ei in ec:
                prod4 *= signs[ei]
            if prod4 == -1:
                return True
        
        return False
    
    def evaluate(signs):
        if not fast_is_connected(signs):
            return float('inf')
        return _spectral_radius_of_signed_adj(graph, signs)
    
    # Generate initial candidates
    initial_candidates = []
    
    # 1. Harmonic initialization
    harmonic = _harmonic_init(graph)
    if fast_is_connected(harmonic):
        initial_candidates.append(harmonic)
    
    # 2. All +1 with one -1 to ensure connectivity
    all_plus = np.ones(num_edges, dtype=np.int8)
    all_plus[0] = -1  # Flip one edge to ensure some cycle has product -1
    if fast_is_connected(all_plus):
        initial_candidates.append(all_plus)
    
    # 3. Random initializations
    for _ in range(20):
        cand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        if fast_is_connected(cand):
            initial_candidates.append(cand)
    
    # 4. Structured patterns: alternate signs on top and bottom cycles
    for pattern in range(4):
        cand = np.ones(num_edges, dtype=np.int8)
        # Top cycle: alternate
        for i in range(m):
            if i % 2 == pattern % 2:
                cand[top_cycle_edges[i]] = -1
        # Bottom cycle: alternate with offset
        for i in range(m):
            if i % 2 == (pattern + 1) % 2:
                cand[bot_cycle_edges[i]] = -1
        if fast_is_connected(cand):
            initial_candidates.append(cand)
    
    # Local search from each initial candidate
    for init_signs in initial_candidates:
        if time.time() > deadline:
            break
        
        signs = init_signs.copy()
        current_radius = evaluate(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
        
        if current_radius == float('inf'):
            continue
        
        # Local search: try all single-edge flips, pick best improvement
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            best_flip = -1
            best_new_radius = current_radius
            
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                
                old_sign = signs[i]
                signs[i] = -old_sign
                new_radius = evaluate(signs)
                signs[i] = old_sign
                
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_flip = i
            
            if best_flip >= 0:
                signs[best_flip] *= -1
                current_radius = best_new_radius
                improved = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = signs.copy()
        
        # 2-edge flip local search
        if time.time() > deadline:
            break
        
        improved2 = True
        while improved2:
            if time.time() > deadline:
                break
            
            improved2 = False
            best_flip_pair = None
            best_new_radius = current_radius
            
            # Try all pairs for small graphs, sample for larger
            num_pairs_to_try = min(200, num_edges * (num_edges - 1) // 2)
            
            for _ in range(num_pairs_to_try):
                if time.time() > deadline:
                    break
                
                i, j = rng.choice(num_edges, size=2, replace=False)
                if i > j:
                    i, j = j, i
                
                si, sj = signs[i], signs[j]
                signs[i] *= -1
                signs[j] *= -1
                new_radius = evaluate(signs)
                signs[i] = si
                signs[j] = sj
                
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_flip_pair = (i, j)
            
            if best_flip_pair is not None:
                i, j = best_flip_pair
                signs[i] *= -1
                signs[j] *= -1
                current_radius = best_new_radius
                improved2 = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = signs.copy()
    
    # If we still have time, do more random restarts
    if best_signs is None or time.time() < deadline - 5:
        num_extra_restarts = 100
        for restart in range(num_extra_restarts):
            if time.time() > deadline:
                break
            
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            
            current_radius = evaluate(signs)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = signs.copy()
            
            if current_radius == float('inf'):
                continue
            
            # Quick local search
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                best_flip = -1
                best_new_radius = current_radius
                
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    new_radius = evaluate(signs)
                    signs[i] = old_sign
                    
                    if new_radius < best_new_radius:
                        best_new_radius = new_radius
                        best_flip = i
                
                if best_flip >= 0:
                    signs[best_flip] *= -1
                    current_radius = best_new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = signs.copy()
                else:
                    break
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not fast_is_connected(best_signs):
            for _ in range(10000):
                cand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                if fast_is_connected(cand):
                    best_signs = cand
                    break
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
