"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_adjacency(graph):
    """Build the unsigned adjacency matrix for the base graph."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A, edges

def _signed_adjacency(A, signs):
    """Apply signs to the adjacency matrix. Signs are per-edge."""
    A_signed = A.copy()
    # We need to map edge index to (u,v)
    # This is inefficient; better to build from edges directly
    return A_signed

def _spectral_radius_signed(graph, signs):
    """Compute the spectral radius of the signed adjacency matrix."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        A[u, v] = s
        A[v, u] = s
    # For a signed graph, the eigenvalues of the signed adjacency matrix
    # are what we want. The spectral radius is the max absolute eigenvalue.
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    except Exception:
        return np.inf

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular.
    Since the base graph is a prism (3-regular), the lift will always be 3-regular
    if it's a valid 2-lift. The connectivity is the key constraint.
    
    Actually, for a signed 2-lift of a connected base graph, the lift is connected
    iff the signs are not all +1 on any cycle that would allow a bipartite decomposition
    into two disconnected components. More precisely, the lift is disconnected iff
    there exists a non-trivial subset of vertices such that all edges between the subset
    and its complement have sign +1 (or all -1, depending on convention).
    
    For a 2-lift defined by signs: vertices are (v,0) and (v,1).
    Edge (u,v) with sign s: connects (u,0)-(v,0) and (u,1)-(v,1) if s=+1,
    or (u,0)-(v,1) and (u,1)-(v,0) if s=-1.
    
    The lift is disconnected iff there is a partition of the base graph vertices into S and T
    such that all edges between S and T have the same sign (all +1 or all -1), AND
    the signs within S and within T are consistent with a 2-coloring that separates the lift.
    
    Actually, the standard criterion: the signed 2-lift is connected iff the sign vector
    is not in the cut space. I.e., there is no non-trivial subset S of V such that
    the sign of every edge crossing the cut (S, V\S) is the same.
    
    Let's just check connectivity of the lift graph directly for small graphs.
    """
    n = graph['n']
    edges = graph['edges']
    # Build the lift graph adjacency list
    # Vertices in lift: 0 to 2*n-1. (v,0) -> v, (v,1) -> v+n
    lift_adj = [[] for _ in range(2 * n)]
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        if s == 1:
            lift_adj[u].append(v)
            lift_adj[v].append(u)
            lift_adj[u + n].append(v + n)
            lift_adj[v + n].append(u + n)
        else:
            lift_adj[u].append(v + n)
            lift_adj[v + n].append(u)
            lift_adj[u + n].append(v)
            lift_adj[v].append(u + n)
    
    # BFS from vertex 0
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return all(visited)

def _try_signs(graph, signs):
    """Return spectral radius if valid, else inf."""
    if not _is_valid_lift(graph, signs):
        return np.inf
    return _spectral_radius_signed(graph, signs)

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    E = len(edges)
    
    # Deadline guard
    deadline = time.time() + 85.0  # 85s out of 90s limit
    
    # Start with random sampling
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Random sampling phase
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=E).astype(np.int64)
        r = _try_signs(graph, signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 might be valid? Check.
        signs = np.ones(E, dtype=np.int64)
        if _is_valid_lift(graph, signs):
            best_signs = signs
            best_radius = _spectral_radius_signed(graph, signs)
        else:
            # Try all -1
            signs = -np.ones(E, dtype=np.int64)
            if _is_valid_lift(graph, signs):
                best_signs = signs
                best_radius = _spectral_radius_signed(graph, signs)
            else:
                # This should not happen for prism graphs, but just in case
                best_signs = np.ones(E, dtype=np.int64)
                best_radius = _spectral_radius_signed(graph, signs)
    
    # Local search phase: single flips and pairwise flips
    current_signs = best_signs.copy()
    current_radius = best_radius
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Single flips
        for i in range(E):
            if time.time() > deadline:
                break
            trial = current_signs.copy()
            trial[i] = -trial[i]
            r = _try_signs(graph, trial)
            if r < current_radius - 1e-12:
                current_signs = trial
                current_radius = r
                best_signs = trial.copy()
                best_radius = r
                improved = True
        
        # Pairwise flips
        for i in range(E):
            if time.time() > deadline:
                break
            for j in range(i + 1, E):
                if time.time() > deadline:
                    break
                trial = current_signs.copy()
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                r = _try_signs(graph, trial)
                if r < current_radius - 1e-12:
                    current_signs = trial
                    current_radius = r
                    best_signs = trial.copy()
                    best_radius = r
                    improved = True
    
    # If we have time, try more aggressive local search or restarts
    if time.time() < deadline - 10:
        # Try a few random restarts with local search
        for restart in range(5):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=E).astype(np.int64)
            r = _try_signs(graph, signs)
            if r < best_radius:
                best_signs = signs.copy()
                best_radius = r
            
            # Local search from this point
            current_signs = best_signs.copy()
            current_radius = best_radius
            local_improved = True
            while local_improved and time.time() < deadline:
                local_improved = False
                for i in range(E):
                    if time.time() > deadline:
                        break
                    trial = current_signs.copy()
                    trial[i] = -trial[i]
                    r = _try_signs(graph, trial)
                    if r < current_radius - 1e-12:
                        current_signs = trial
                        current_radius = r
                        best_signs = trial.copy()
                        best_radius = r
                        local_improved = True
    
    return best_signs
# EVOLVE-BLOCK-END
