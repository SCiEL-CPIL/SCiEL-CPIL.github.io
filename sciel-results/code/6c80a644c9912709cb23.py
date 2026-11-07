"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline with 15s safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)
    
    # Identify edge types: cycle edges and matching edges
    # Cycle edges: (i, i+1) or (i+1, i) for i in 0..m-1 (with wrap)
    # Matching edges: (i, i+m)
    
    # Build adjacency for quick lookup
    # For prism graph, edges are:
    # Top cycle: (0,1), (1,2), ..., (m-2, m-1), (m-1, 0)
    # Bottom cycle: (m, m+1), ..., (2m-2, 2m-1), (2m-1, m)
    # Matching: (0, m), (1, m+1), ..., (m-1, 2m-1)
    
    # We need to assign signs to all edges
    # Key insight: gauge invariance means we can fix some signs
    # A gauge transformation flips signs of all edges incident to a vertex
    # This means the product of signs around any cycle is invariant
    # 
    # For the prism graph, there are m triangular faces (each involving two cycle edges and one matching edge)
    # plus the two m-cycles
    # 
    # The spectral radius depends only on the cycle products (fluxes)
    # We can use gauge freedom to simplify
    
    # Let's use a systematic approach:
    # Fix the gauge by setting all matching edges to +1
    # Then the cycle edges determine the fluxes
    
    # Actually, let's think more carefully about the structure.
    # The prism graph has 3m edges: m top cycle, m bottom cycle, m matching
    
    # Gauge invariance: flipping all signs incident to vertex v leaves the lift isomorphic
    # This means we can set one vertex's incident edges to reference values
    
    # Strategy: enumerate gauge-equivalent classes
    # Fix matching edge 0 to +1 (gauge choice)
    # Then enumerate remaining signs with constraint
    
    # Better: use the fact that the signed adjacency matrix eigenvalues
    # depend on the "winding numbers" or fluxes around fundamental cycles
    
    # For a prism graph, we can decompose the sign pattern into:
    # - Signs on top cycle: t_0, ..., t_{m-1}
    # - Signs on bottom cycle: b_0, ..., b_{m-1}  
    # - Signs on matching: s_0, ..., s_{m-1}
    
    # The product around each triangular face (i, i+1, i+1+m, i+m) gives a flux
    # Face i: t_i * s_{i+1} * b_{i+1} * s_i (going around the triangle)
    # Wait, let me be more careful about the cycle orientation
    
    # Let me just do a smart search with gauge fixing
    
    # Gauge fix: set all matching edges to +1
    # This uses m-1 degrees of gauge freedom (one per vertex in one cycle)
    # Actually, gauge has m-1 independent degrees of freedom for 2m vertices
    
    # With matching fixed to +1, we need to assign signs to 2m cycle edges
    # But gauge still has freedom... let me think again
    
    # Gauge transformation: for each vertex v, flip all incident edge signs
    # This is equivalent to: sign_e -> sign_e * (-1)^{f(u) + f(v)} where f is vertex labeling
    # f: V -> {0,1}, and flipping f by a constant doesn't change anything
    # So gauge has 2^(2m-1) elements, but acts on 3m edge signs
    
    # The invariant quantities are products around cycles
    # Fundamental cycles: m triangles + 1 remaining (the "outer" cycle)
    # Actually for a prism, the cycle space has dimension E - V + 1 = 3m - 2m + 1 = m + 1
    
    # So there are m+1 independent cycle fluxes
    # With gauge fixing, we can reduce the search space
    
    # Practical approach: 
    # 1. Fix gauge: set top vertex 0 to f=0, and use it to normalize
    # 2. Enumerate flux patterns around the m triangular faces
    # 3. For each flux pattern, compute the spectral radius
    
    # Let me implement a direct but efficient search
    
    # For small m (4,5,7,9), we can be clever
    # The key invariant is the product of signs around each face
    
    # Let me parameterize by face fluxes
    # Face i (triangle involving vertices i, i+1, i+m, i+1+m):
    #   flux_i = sign(top_i) * sign(match_{i+1}) * sign(bottom_{i+1}) * sign(match_i)
    #   where top_i is edge (i, i+1), match_i is edge (i, i+m)
    #   Note: indices mod m
    
    # Actually, let me just do a local search / random restart approach
    # with gauge invariance exploited for speedup
    
    # Build edge index map
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_map[(u, v)] = idx
        edge_map[(v, u)] = idx
    
    # Identify which edges are matching, top cycle, bottom cycle
    matching_edges = []
    top_cycle_edges = []
    bottom_cycle_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if v == u + m or u == v + m:
            matching_edges.append(idx)
        elif u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
    
    # Sort them for consistency
    matching_edges.sort()
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    
    # Gauge fixing strategy:
    # Set all matching edges to +1
    # Then we only need to assign signs to 2m cycle edges
    # But gauge still allows flipping all edges incident to any vertex
    # With matching fixed, this constrains the cycle edge signs
    
    # Actually, let's use a different approach:
    # The spectral radius of the signed adjacency matrix
    # For a prism graph with signs, we can use the structure to compute eigenvalues efficiently
    
    # The signed adjacency matrix A has the form:
    # A = [[A_top, S], [S^T, A_bottom]]
    # where A_top is the signed top cycle, A_bottom is signed bottom cycle,
    # and S is the signed matching (diagonal if matching is (i, i+m))
    
    # By symmetry, we can block-diagonalize using the "even" and "odd" subspaces
    # A acting on (x_top, x_bottom):
    #   (A_top x_top + S x_bottom, S^T x_top + A_bottom x_bottom)
    # 
    # Even subspace: x_top = x_bottom -> (A_top + S)x, (S^T + A_bottom)x
    # Odd subspace: x_top = -x_bottom -> (A_top - S)x, (-S^T + A_bottom)x
    
    # So eigenvalues are union of eigenvalues of (A_top + S) and (A_bottom + S^T)
    # Wait, that's not quite right. Let me redo.
    
    # A = [[A_top, M], [M^T, A_bottom]] where M is the matching sign matrix (diagonal)
    # Eigenvector (x, y): A_top x + M y = λ x, M^T x + A_bottom y = λ y
    # 
    # If we assume M is diagonal with entries s_i (matching signs), then:
    # From first eq: y = M^{-1}(λ I - A_top) x = D(λ I - A_top) x where D = M^{-1} = M (since M^2 = I)
    # Substitute: M^T x + A_bottom D (λ I - A_top) x = λ x
    # M x + A_bottom M (λ I - A_top) x = λ x  [since M^T = M for diagonal]
    # M x + A_bottom M λ x - A_bottom M A_top x = λ x
    # This is getting complicated. Let me just use eigvalsh directly.
    
    # For efficiency, I'll build the matrix and use eigvalsh
    # For m up to 9, n=18, so 18x18 matrix - very fast
    
    def build_signed_adjacency(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] += s
            A[v, u] += s
        return A
    
    def compute_radius(signs):
        A = build_signed_adjacency(signs)
        eigs = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigs))
    
    def is_connected(signs):
        # Check if the lift is connected
        # Lift has 2n vertices: (v, 0) and (v, 1)
        # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency list for lift
        lift_adj = [[] for _ in range(2*n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u*2].append(v*2)
                lift_adj[v*2].append(u*2)
                lift_adj[u*2+1].append(v*2+1)
                lift_adj[v*2+1].append(u*2+1)
            else:
                lift_adj[u*2].append(v*2+1)
                lift_adj[v*2+1].append(u*2)
                lift_adj[u*2+1].append(v*2)
                lift_adj[v*2].append(u*2+1)
        
        # BFS from vertex 0
        visited = [False] * (2*n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == 2*n
    
    # Search strategy:
    # 1. Start with all +1 (trivially connected, spectral radius = max degree = 3)
    # 2. Use local search: flip individual signs to reduce spectral radius
    # 3. Use random restarts with gauge-equivalent random starts
    
    best_signs = None
    best_radius = float('inf')
    
    # Initial: all +1
    init_signs = np.ones(E, dtype=np.int8)
    if is_connected(init_signs):
        r = compute_radius(init_signs)
        best_signs = init_signs.copy()
        best_radius = r
    
    # Local search: try flipping each sign
    def local_search(signs, max_iters=10000):
        nonlocal best_signs, best_radius
        current = signs.copy()
        current_radius = compute_radius(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try flipping each edge
            for i in range(E):
                current[i] = -current[i]
                if is_connected(current):
                    r = compute_radius(current)
                    if r < current_radius - 1e-12:
                        current_radius = r
                        improved = True
                        if r < best_radius:
                            best_radius = r
                            best_signs = current.copy()
                        break
                    else:
                        current[i] = -current[i]  # revert
                else:
                    current[i] = -current[i]  # revert
            
            if not improved:
                break
        return current
    
    local_search(best_signs)
    
    # Random restarts with gauge-equivalent patterns
    rng = np.random.default_rng(42)
    
    for restart in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Random sign pattern
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        
        if not is_connected(signs):
            continue
        
        r = compute_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        
        # Local search from this start
        result = local_search(signs, max_iters=5000)
        if result is not None:
            r = compute_radius(result)
            if r < best_radius:
                best_radius = r
                best_signs = result.copy()
    
    # Final polish: more aggressive local search
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Try 2-flips
            improved = False
            for i in range(E):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, E):
                    if time.time() > deadline - 5:
                        break
                    best_signs[i] = -best_signs[i]
                    best_signs[j] = -best_signs[j]
                    if is_connected(best_signs):
                        r = compute_radius(best_signs)
                        if r < best_radius - 1e-12:
                            best_radius = r
                            improved = True
                            break
                        else:
                            best_signs[i] = -best_signs[i]
                            best_signs[j] = -best_signs[j]
                    else:
                        best_signs[i] = -best_signs[i]
                        best_signs[j] = -best_signs[j]
                if improved:
                    break
            if not improved:
                break
    
    return best_signs
# EVOLVE-BLOCK-END
