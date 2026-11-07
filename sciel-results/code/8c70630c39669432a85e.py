def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85 # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Precompute edge endpoints as arrays for fast access
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Build base adjacency matrix structure
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius_from_A(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))
    
    # Fast validity check: for a 2-lift of a connected graph,
    # the lift is connected iff the product of signs around some
    # fundamental cycle is -1. For prism graphs (two m-cycles + matching),
    # we can check: the holonomy around the "top" cycle, or equivalently,
    # the product of all signs must be -1 for connectivity.
    # Actually, let's verify: the lift has 2n vertices. It's connected iff
    # there's no bipartition invariant. The key is that the sign product
    # around the cycle that goes up, across, and down must be -1.
    # For prism: take edge (0, m) [matching], then cycle on top 0->1->...->m-1->0,
    # then matching (m-1, 2m-1). The product of signs on this closed walk determines connectivity.
    # More simply: just check if product of all edge signs is -1.
    # Let me verify this is correct for prism graphs...
    # Actually, the safest fast check: the 2-lift of a connected graph is connected
    # iff the sign vector is not "balanced" (i.e., there's no vertex labeling s:V->{-1,1}
    # such that for every edge (u,v), sign(u,v) = s[u]*s[v]).
    # For a tree, any signing is unbalanced. For graphs with cycles, it depends on cycle products.
    # For efficiency, let's just do a quick BFS-based check but with early termination.
    
    def is_valid_lift_fast(signs):
        # Quick check: product of all signs
        # For prism graphs, this should work as a fast filter
        prod = np.prod(signs)
        if prod == 1:
            # Likely disconnected, but let's verify with a quick check
            # Actually for these specific graphs, let's just do the BFS
            pass
        # Do a lightweight BFS
        adj = [[] for _ in range(2 * n_vertices)]
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            s = signs[i]
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
        
        visited = np.zeros(2 * n_vertices, dtype=bool)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    if count == 2 * n_vertices:
                        return True
                    stack.append(nb)
        return count == 2 * n_vertices

    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    # Pre-build initial adjacency
    def evaluate_signs(signs):
        A = build_signed_adj(signs)
        radius = spectral_radius_from_A(A)
        return radius
    
    # Multiple restarts
    for restart in range(150):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Initialize
        if restart == 0:
            # Start with a known valid configuration: flip one edge from all +1
            signs = np.ones(n_edges, dtype=int)
            signs[0] = -1  # Make it connected
        elif restart == 1:
            signs = rng.choice([-1, 1], size=n_edges)
        else:
            signs = best_signs.copy() if best_signs is not None else rng.choice([-1, 1], size=n_edges)
            # Perturb
            n_flips = rng.integers(1, 4)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            signs[idxs] *= -1

        # Local Search
        current_signs = signs.copy()
        
        # Check validity first
        if not is_valid_lift_fast(current_signs):
            continue
            
        current_A = build_signed_adj(current_signs)
        current_radius = spectral_radius_from_A(current_A)
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            # Try 1-flip
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1: break
                old = current_signs[i]
                current_signs[i] = -old
                
                # Update A in-place
                u, v = edge_u[i], edge_v[i]
                current_A[u, v] = current_signs[i]
                current_A[v, u] = current_signs[i]
                
                new_radius = spectral_radius_from_A(current_A)
                
                if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old
                    current_A[u, v] = old
                    current_A[v, u] = old
                    
            # If no 1-flip improved, try 2-flips (stochastic)
            if not improved:
                # Try a few random 2-flips
                for _ in range(20):
                    if time.time() - start_time > DEADLINE - 1: break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    oi = current_signs[i]
                    oj = current_signs[j]
                    current_signs[i] = -oi
                    current_signs[j] = -oj
                    
                    # Update A in-place
                    ui, vi = edge_u[i], edge_v[i]
                    uj, vj = edge_u[j], edge_v[j]
                    current_A[ui, vi] = current_signs[i]
                    current_A[vi, ui] = current_signs[i]
                    current_A[uj, vj] = current_signs[j]
                    current_A[vj, uj] = current_signs[j]
                    
                    new_radius = spectral_radius_from_A(current_A)
                    
                    if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                        current_radius = new_radius
                        improved = True
                    else:
                        current_signs[i] = oi
                        current_signs[j] = oj
                        current_A[ui, vi] = oi
                        current_A[vi, ui] = oi
                        current_A[uj, vj] = oj
                        current_A[vj, uj] = oj
                        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    if best_signs is None:
        # Fallback: find a valid configuration
        best_signs = rng.choice([-1, 1], size=n_edges)
        for _ in range(1000):
            if is_valid_lift_fast(best_signs):
                break
            idx = rng.integers(0, n_edges)
            best_signs[idx] *= -1

    return best_signs
