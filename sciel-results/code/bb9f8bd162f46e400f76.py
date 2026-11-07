def sign_edges(graph, budget):
    import numpy as np
    import time

    start_time = time.time()
    DEADLINE = start_time + 80.0  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Classify edges by role
    top_edges = []
    bot_edges = []
    match_edges = []
    for idx, (u, v) in enumerate(edges):
        u, v = int(u), int(v)
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bot_edges.append(idx)
        else:
            match_edges.append(idx)

    # Precompute edge endpoints as arrays for speed
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A

    def spectral_radius(signs):
        A = build_signed_adj(signs)
        vals = np.linalg.eigvalsh(A)
        return np.max(np.abs(vals))

    def is_valid_lift(signs):
        """Check if the 2-lift is connected using holonomy/cycle products."""
        # The lift is connected iff there is no non-trivial subset of vertices
        # that is invariant under the lift edges. For a prism graph (two cycles + matching),
        # the fundamental cycles are:
        # 1. Top cycle
        # 2. Bottom cycle
        # 3. The "ladder" cycle formed by going along top, down a matching, back along bottom, up matching.
        
        # A 2-lift is disconnected iff there exists a labeling of vertices by {0,1} 
        # (up to global flip) such that for every edge (u,v) with sign s,
        # label[u] XOR label[v] == (0 if s==1 else 1).
        # This is a system of linear equations over GF(2).
        # The lift is connected iff this system has exactly 2 solutions (the two connected components).
        # Equivalently, the cycle space constraints must be consistent.
        
        # More directly: build the lift graph and BFS.
        # Since n_vertices is small (max 18), BFS on 36 nodes is very fast.
        adj = [[] for _ in range(2 * n_vertices)]
        for idx in range(n_edges):
            u = edge_u[idx]
            v = edge_v[idx]
            s = signs[idx]
            if s == 1:
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)
        
        visited = [False] * (2 * n_vertices)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == 2 * n_vertices

    def local_search(initial_signs, time_limit=0.5):
        signs = initial_signs.copy()
        current_radius = spectral_radius(signs)
        if not is_valid_lift(signs):
            return None, current_radius
        
        improved = True
        while improved and (time.time() < DEADLINE - 1.0):
            improved = False
            for i in range(n_edges):
                if (time.time() > DEADLINE - 1.0):
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                new_radius = spectral_radius(signs)
                if new_radius < current_radius - 1e-10 and is_valid_lift(signs):
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old_sign
        return signs, current_radius

    # Strategy: Enumerate all 2^(m+1) switching classes if feasible, otherwise use local search.
    # The key insight is that the spectrum is invariant under switching (gauge transformation).
    # We can fix the signs of a spanning tree to +1 and only vary the non-tree edges.
    # For a prism graph, we can choose a spanning tree that includes all matching edges and
    # m-1 edges from each cycle, leaving 2 cycle-closing edges and (m-2) matching edges as non-tree?
    # Actually, let's just use the approach from inspiration 1: identify non-tree edges.
    
    # Build a spanning tree:
    # - Include all matching edges: (i, i+m) for i=0..m-1. That's m edges.
    # - Include m-1 edges from the top cycle: (0,1), (1,2), ..., (m-2, m-1).
    # - Include m-1 edges from the bottom cycle: (m, m+1), ..., (2m-2, 2m-1).
    # Total tree edges: m + (m-1) + (m-1) = 3m - 2.
    # Total edges: 3m.
    # Non-tree edges: 3m - (3m-2) = 2.
    # Wait, the prism graph has 3m edges.
    # Let's re-verify:
    # Top cycle: m edges.
    # Bottom cycle: m edges.
    # Matching: m edges.
    # Total: 3m.
    
    # Let's identify the non-tree edges.
    # Tree edges:
    # Matching: all m.
    # Top path: (i, i+1) for i=0..m-2. (m-1 edges)
    # Bottom path: (m+i, m+i+1) for i=0..m-2. (m-1 edges)
    # Non-tree edges:
    # Top close: (m-1, 0)
    # Bottom close: (2m-1, m)
    # Total non-tree: 2.
    
    # So we only need to enumerate 2^2 = 4 combinations?
    # But the spectrum depends on the product of signs around cycles.
    # The fundamental cycles are:
    # 1. Top cycle: product of top edge signs.
    # 2. Bottom cycle: product of bottom edge signs.
    # 3. Ladder cycle: product of top path, matching, bottom path (reversed), matching.
    
    # Actually, the switching invariance means we can fix all tree edges to +1.
    # Then the signs of the non-tree edges determine the holonomy.
    # Let's find the indices of the non-tree edges.
    
    non_tree_indices = []
    for idx, (u, v) in enumerate(edges):
        u, v = int(u), int(v)
        # Check if it's a tree edge
        is_tree = False
        if abs(u - v) == m:
            is_tree = True  # All matching edges are in the tree
        elif u < m and v < m:
            # Top cycle
            if abs(u - v) == 1:
                is_tree = True
            else:
                is_tree = False  # This is the closing edge (m-1, 0)
        elif u >= m and v >= m:
            # Bottom cycle
            if abs(u - v) == 1:
                is_tree = True
            else:
                is_tree = False  # This is the closing edge (2m-1, m)
        
        if not is_tree:
            non_tree_indices.append(idx)
    
    # Now enumerate all 2^len(non_tree_indices) combinations
    # Set tree edges to +1, vary non-tree edges.
    
    best_signs = None
    best_radius = float('inf')
    
    num_non_tree = len(non_tree_indices)
    total_combinations = 1 << num_non_tree
    
    for mask in range(total_combinations):
        if time.time() > DEADLINE - 2.0:
            break
        
        signs = np.ones(n_edges, dtype=int)
        for j, idx in enumerate(non_tree_indices):
            if (mask >> j) & 1:
                signs[idx] = -1
        
        if not is_valid_lift(signs):
            continue
        
        radius = spectral_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If enumeration didn't find a valid lift (shouldn't happen), or if we want to improve further,
    # use local search on the best found so far.
    if best_signs is not None:
        s, r = local_search(best_signs)
        if s is not None and r < best_radius:
            best_signs = s
            best_radius = r
    else:
        # Fallback: try all +1
        signs = np.ones(n_edges, dtype=int)
        if is_valid_lift(signs):
            best_signs = signs
            best_radius = spectral_radius(signs)
        else:
            # Try a simple heuristic
            signs = np.ones(n_edges, dtype=int)
            for i, (u, v) in enumerate(edges):
                u, v = int(u), int(v)
                if abs(u - v) == m:
                    top = u if u < m else v
                    if top % 2 == 0:
                        signs[i] = -1
            if is_valid_lift(signs):
                best_signs = signs
                best_radius = spectral_radius(signs)
    
    # If we still have time, do a few random restarts with local search
    if time.time() < DEADLINE - 5.0 and best_signs is not None:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() > DEADLINE - 2.0:
                break
            
            # Perturb the best solution
            signs = best_signs.copy()
            k = rng.integers(1, max(2, n_edges // 4))
            flip_idx = rng.choice(n_edges, size=k, replace=False)
            signs[flip_idx] *= -1
            
            s, r = local_search(signs)
            if s is not None and r < best_radius:
                best_signs = s
                best_radius = r
    
    if best_signs is None:
        # Absolute fallback
        best_signs = np.ones(n_edges, dtype=int)
    
    return best_signs
