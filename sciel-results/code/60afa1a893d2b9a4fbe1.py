"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_adjacency(graph, signs):
    """Build the signed adjacency matrix for the base graph."""
    n = graph['n']
    A = np.zeros((n, n))
    for (u, v), s in zip(graph['edges'], signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _matching_signs(graph, signs):
    """Extract signs on matching edges (i, i+m) for i in 0..m-1."""
    n = graph['n']
    m = n // 2
    d = np.ones(m)
    for (u, v), s in zip(graph['edges'], signs):
        if abs(u - v) == m:
            i = min(u, v)
            d[i] = s
    return d

def _spectral_radius(A):
    """Compute spectral radius (max absolute eigenvalue) of symmetric matrix."""
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    except np.linalg.LinAlgError:
        return np.inf

def _lift_spectral_radius(graph, signs):
    """Compute spectral radius of the 2-lift."""
    n = graph['n']
    m = n // 2
    A = _build_adjacency(graph, signs)
    d = _matching_signs(graph, signs)
    
    # The lift's eigenvalues are eigenvalues of A and eigenvalues of D*A
    # where D = diag(d) on the matching
    # Actually, the 2-lift adjacency is:
    # [[A, D], [D, A]] where D is the matching part
    # Eigenvalues are eig(A + D) and eig(A - D)
    
    # Wait, let me reconsider. The prism graph has:
    # - Top cycle edges: (i, i+1) for i in 0..m-1 (mod m)
    # - Bottom cycle edges: (i+m, i+1+m) for i in 0..m-1 (mod m)  
    # - Matching edges: (i, i+m) for i in 0..m-1
    
    # The signed adjacency A has signs on all edges.
    # The 2-lift adjacency matrix B is 2n x 2n:
    # B = [[A_top, M], [M, A_bottom]]
    # where A_top is the top cycle adjacency (m x m), A_bottom is bottom cycle
    # and M is the matching with signs.
    
    # Actually, for the standard 2-lift construction:
    # Vertices: (v, 0) and (v, 1) for each base vertex v
    # Edge (u,v) with sign s: connects (u,0)-(v,0) if s=+1, else (u,0)-(v,1)
    # and (u,1)-(v,1) if s=+1, else (u,1)-(v,0)
    
    # So the lift adjacency B satisfies:
    # B[(u,i),(v,j)] = 1 if there's edge (u,v) in base and j = i if s=+1, j=1-i if s=-1
    # This means B = [[A, S], [S, A]] where A is the base adjacency and
    # S is... hmm, this isn't quite right for non-bipartite.
    
    # Let me just construct it directly.
    B = np.zeros((2*n, 2*n))
    for (u, v), s in zip(graph['edges'], signs):
        if s == 1:
            B[u, v] = 1
            B[v, u] = 1
            B[u+n, v+n] = 1
            B[v+n, u+n] = 1
        else:
            B[u, v+n] = 1
            B[v, u+n] = 1
            B[u+n, v] = 1
            B[v+n, u] = 1
    
    return _spectral_radius(B)

def _is_connected_lift(graph, signs):
    """Check if the 2-lift is connected."""
    n = graph['n']
    num_vertices = 2 * n
    
    # Build adjacency list for the lift
    adj = [[] for _ in range(num_vertices)]
    for (u, v), s in zip(graph['edges'], signs):
        if s == 1:
            adj[u].append(v)
            adj[v].append(u)
            adj[u+n].append(v+n)
            adj[v+n].append(u+n)
        else:
            adj[u].append(v+n)
            adj[v].append(u+n)
            adj[u+n].append(v)
            adj[v+n].append(u)
    
    # BFS from vertex 0
    visited = [False] * num_vertices
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == num_vertices

def _evaluate(graph, signs):
    """Return spectral radius if valid, else inf."""
    if not _is_connected_lift(graph, signs):
        return np.inf
    return _lift_spectral_radius(graph, signs)

def sign_edges(graph, budget):
    """Find signs minimizing the spectral radius of the 2-lift."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = graph['n']
    m = n // 2
    num_edges = len(graph['edges'])
    
    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Precompute edge info
    edges = graph['edges']
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        r = _evaluate(graph, signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Phase 1: Random sampling
    num_random = 2000
    for i in range(num_random):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        try_signs(signs)
    
    # Phase 2: Local search from best and some random starts
    def local_search(start_signs, max_iters=500):
        nonlocal best_signs, best_radius
        signs = start_signs.copy()
        current_radius = _evaluate(graph, signs)
        if np.isinf(current_radius):
            # Try to fix connectivity
            for _ in range(100):
                idx = rng.integers(0, num_edges)
                signs[idx] = -signs[idx]
                r = _evaluate(graph, signs)
                if not np.isinf(r):
                    current_radius = r
                    break
            if np.isinf(current_radius):
                return
    
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(num_edges):
                if time.time() > deadline:
                    return
                old = signs[idx]
                signs[idx] = -old
                new_radius = _evaluate(graph, signs)
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old
    
    # Local search from best
    if best_signs is not None:
        local_search(best_signs)
    
    # Local search from several random starts
    for i in range(50):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        local_search(signs)
    
    # Phase 3: Simulated annealing-like refinement
    if time.time() < deadline:
        signs = best_signs.copy()
        current_radius = _evaluate(graph, signs)
        if not np.isinf(current_radius):
            temp = 0.01
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                idx = rng.integers(0, num_edges)
                signs[idx] = -signs[idx]
                new_radius = _evaluate(graph, signs)
                if new_radius < current_radius:
                    current_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = signs.copy()
                else:
                    delta = new_radius - current_radius
                    if delta < 0:
                        # Accept
                        current_radius = new_radius
                    elif rng.random() < np.exp(-delta / temp):
                        current_radius = new_radius
                    else:
                        signs[idx] = -signs[idx]  # revert
                temp *= 0.995
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
