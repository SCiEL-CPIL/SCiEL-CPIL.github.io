"""Spectral graph lifts: minimize spectral radius of signed adjacency matrix."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix for the base graph."""
    n = graph['n']
    edges = graph['edges']
    rows = []
    cols = []
    vals = []
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([s, s])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A


def _spectral_radius(A):
    """Compute spectral radius (max absolute eigenvalue) of symmetric matrix."""
    n = A.shape[0]
    if n <= 1:
        return 0.0
    # Use eigsh for largest magnitude eigenvalues
    try:
        # Get eigenvalues with largest magnitude
        if n <= 20:
            w = np.linalg.eigvalsh(A.toarray())
            return float(np.max(np.abs(w)))
        else:
            # For larger matrices, use eigsh
            # We want the eigenvalue with largest absolute value
            # Try both positive and negative
            w_pos = eigsh(A, k=1, sigma=0, which='LM', return_eigenvectors=False)
            w_neg = eigsh(A, k=1, sigma=0, which='LM', return_eigenvectors=False)
            # Actually for symmetric, just get largest magnitude
            w = eigsh(A, k=min(3, n-1), which='LM', return_eigenvectors=False)
            return float(np.max(np.abs(w)))
    except Exception:
        w = np.linalg.eigvalsh(A.toarray())
        return float(np.max(np.abs(w)))


def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular.
    
    The lift has 2n vertices. For each edge (u,v) with sign s:
    - If s=+1: edges (u,0)-(v,0) and (u,1)-(v,1)
    - If s=-1: edges (u,0)-(v,1) and (u,1)-(v,0)
    
    We need to check connectivity of the lift graph.
    """
    n = graph['n']
    edges = graph['edges']
    num_lift_vertices = 2 * n
    
    # Build adjacency list for the lift
    adj = [[] for _ in range(num_lift_vertices)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    # BFS to check connectivity
    visited = [False] * num_lift_vertices
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == num_lift_vertices


def _local_search(graph, signs, max_iter=5000):
    """Local search: flip one sign at a time to reduce spectral radius."""
    n_edges = len(graph['edges'])
    current_signs = signs.copy()
    current_radius = _spectral_radius(_build_signed_adjacency(graph, current_signs))
    
    if not _is_valid_lift(graph, current_signs):
        return None, float('inf')
    
    best_signs = current_signs.copy()
    best_radius = current_radius
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n_edges):
            # Try flipping sign i
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if not _is_valid_lift(graph, trial):
                continue
            
            A = _build_signed_adjacency(graph, trial)
            radius = _spectral_radius(A)
            
            if radius < best_radius - 1e-12:
                current_signs = trial
                best_signs = trial.copy()
                best_radius = radius
                improved = True
                break  # Restart from new position
        
        if not improved:
            break
    
    return best_signs, best_radius


def _structured_init(m, pattern='alternating'):
    """Generate structured initial sign assignments."""
    # Prism graph: m top cycle edges, m bottom cycle edges, m matching edges
    # Edge order: typically (i, i+1) for i in 0..m-1 (top), (m+i, m+i+1) for bottom, (i, i+m) for matching
    # Let's assume edges are ordered as: top cycle, bottom cycle, matching
    
    signs = np.ones(3 * m, dtype=int)
    
    if pattern == 'alternating':
        # Alternate signs on cycles
        for i in range(m):
            signs[i] = 1 if i % 2 == 0 else -1
            signs[m + i] = 1 if i % 2 == 0 else -1
        # Matching edges: all +1
        for i in range(m):
            signs[2 * m + i] = 1
    elif pattern == 'all_plus':
        signs[:] = 1
    elif pattern == 'all_minus':
        signs[:] = -1
    elif pattern == 'mixed':
        # Top all +1, bottom all -1, matching +1
        signs[:m] = 1
        signs[m:2*m] = -1
        signs[2*m:] = 1
    
    return signs


def sign_edges(graph, budget):
    """Find sign assignments minimizing spectral radius of signed adjacency matrix."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    best_signs = None
    best_radius = float('inf')
    
    # Try multiple restarts with different initializations
    patterns = ['alternating', 'all_plus', 'all_minus', 'mixed']
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        
        init_signs = _structured_init(m, pattern)
        
        # Ensure valid lift
        if not _is_valid_lift(graph, init_signs):
            # Try flipping some signs to make it valid
            found_valid = False
            for attempt in range(50):
                trial = init_signs.copy()
                idx = np.random.default_rng(attempt).integers(0, n_edges)
                trial[idx] = -trial[idx]
                if _is_valid_lift(graph, trial):
                    init_signs = trial
                    found_valid = True
                    break
            if not found_valid:
                continue
        
        signs, radius = _local_search(graph, init_signs, max_iter=3000)
        
        if signs is not None and radius < best_radius:
            best_radius = radius
            best_signs = signs
    
    # If no valid solution found, fall back to random search
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            if _is_valid_lift(graph, signs):
                radius = _spectral_radius(_build_signed_adjacency(graph, signs))
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs
    
    if best_signs is None:
        # Last resort: all +1 (should be valid for prism graphs)
        best_signs = np.ones(n_edges, dtype=int)
    
    return best_signs
