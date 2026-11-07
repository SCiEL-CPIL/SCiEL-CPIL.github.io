"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit (evaluator kills at 90s)
    
    # Identify edge types
    # Edges are ordered: top cycle (m), bottom cycle (m), matching (m)
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    # Build edge index map for quick lookup
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Spanning tree for gauge fixing:
    # Tree edges: all m matching edges + top_edges[0..m-2] (m-1 edges)
    # Total: 2m-1 edges (correct for spanning tree of 2m vertices)
    # Chord edges: top_edges[m-1] + all m bottom_edges
    # Total chords: m+1
    
    tree_edge_indices = match_edges + top_edges[:m-1]
    chord_edge_indices = [top_edges[m-1]] + bottom_edges
    num_chords = len(chord_edge_indices)
    
    def build_signed_adj(signs):
        """Build the signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def compute_radius(A):
        """Compute spectral radius from eigenvalues."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except Exception:
            return float('inf')
    
    def check_connectivity(signs):
        """Check if the 2-lift is connected using BFS."""
        # Build adjacency for the lift
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
        # Total lift vertices: 2n = 4m
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # +1 joins equal sides
                adj[u*2].append(v*2)
                adj[v*2].append(u*2)
                adj[u*2+1].append(v*2+1)
                adj[v*2+1].append(u*2+1)
            else:
                # -1 joins opposite sides
                adj[u*2].append(v*2+1)
                adj[v*2+1].append(u*2)
                adj[u*2+1].append(v*2)
                adj[v*2].append(u*2+1)
        
        # BFS from vertex 0
        visited = [False] * lift_n
        visited[0] = True
        queue = [0]
        count = 1
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        
        return count == lift_n
    
    def evaluate(signs):
        """Evaluate a sign assignment: return spectral radius if valid, else inf."""
        # Check connectivity first (cheaper than eigvalsh for invalid cases)
        if not check_connectivity(signs):
            return float('inf')
        
        A = build_signed_adj(signs)
        return compute_radius(A)
    
    def gauge_fix(signs):
        """Apply gauge transformation to make all tree edges +1."""
        signs = np.array(signs, dtype=int, copy=True)
        
        # vertex_sign[v] is the gauge factor for vertex v
        vertex_sign = np.ones(n, dtype=int)
        
        # Propagate through matching edges: (i, i+m)
        for i in range(m):
            vertex_sign[i+m] = vertex_sign[i] * signs[match_edges[i]]
        
        # Propagate through top cycle edges: (i, i+1) for i=0..m-2
        for i in range(m-1):
            vertex_sign[i+1] = vertex_sign[i] * signs[top_edges[i]]
        
        # Apply gauge transformation
        for i, (u, v) in enumerate(edges):
            signs[i] = vertex_sign[u] * vertex_sign[v] * signs[i]
        
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        return r
    
    # Phase 1: Exhaustive search over all 2^(m+1) chord assignments
    # For m=4: 32, m=5: 64, m=7: 256, m=9: 1024 -- all trivially fast
    for mask in range(2 ** num_chords):
        if time.time() > deadline:
            break
        
        # Create sign vector with all +1, then set chord edges based on mask
        base_signs = np.ones(num_edges, dtype=int)
        for j, edge_idx in enumerate(chord_edge_indices):
            if (mask >> j) & 1:
                base_signs[edge_idx] = -1
        
        # Gauge fix
        signs = gauge_fix(base_signs)
        try_signs(signs)
    
    # Phase 2: Local search from the best solution found
    if best_signs is not None:
        current = best_signs.copy()
        current_radius = best_radius
        
        # Try flipping single edges
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                trial = current.copy()
                trial[i] *= -1
                r = evaluate(trial)
                if r < current_radius - 1e-12:
                    current = trial
                    current_radius = r
                    improved = True
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current.copy()
        
        # Try flipping pairs of edges
        if time.time() < deadline - 5:
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i+1, num_edges):
                    if time.time() > deadline:
                        break
                    trial = best_signs.copy()
                    trial[i] *= -1
                    trial[j] *= -1
                    r = evaluate(trial)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        best_signs = trial.copy()
    
    # Phase 3: If we still have time, do random restarts with local search
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random sign assignment
            signs = rng.choice([-1, 1], size=num_edges).astype(int)
            
            # Quick local search: flip single edges
            r = evaluate(signs)
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(num_edges):
                    trial = signs.copy()
                    trial[i] *= -1
                    r2 = evaluate(trial)
                    if r2 < r - 1e-12:
                        signs = trial
                        r = r2
                        improved = True
                        break
                if not improved:
                    break
            
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
