"""Spectral graph lifts: search-based sign assignment for prism graphs."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_base_graph(graph):
    """Precompute adjacency list and edge mapping."""
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    # Build adjacency: for each edge index, store (u, v)
    # Also build a mapping from (u,v) or (v,u) to edge index for quick lookup
    return n, edges, num_edges

def _evaluate_signs(n, edges, signs):
    """
    Compute the spectral radius of the signed adjacency matrix.
    Also check validity: the lift must be connected and 3-regular.
    
    The signed adjacency matrix A has A[u,v] = signs[i] if edge i connects u and v.
    The eigenvalues of the lift are related to eigenvalues of A.
    Specifically, for a 2-lift with signs, the spectrum of the lift's adjacency matrix
    is the union of spectra of A and D*A where D is a diagonal matrix that flips signs 
    on one layer. But the task says: "spectral radius of the signed adjacency matrix 
    of the base graph, max |eigenvalue| from np.linalg.eigvalsh. These are the new 
    eigenvalues of the lift."
    
    Actually, re-reading: "scoring: mean over the four graphs of the spectral radius 
    of the signed adjacency matrix of the base graph, max |eigenvalue| from 
    np.linalg.eigvalsh. These are the new eigenvalues of the lift."
    
    So we just need to compute the spectral radius of the signed adjacency matrix A 
    (which is a 2m x 2m symmetric matrix). The eigenvalues of A are what we report.
    
    For validity: the lift must be connected and 3-regular.
    - 3-regular: each vertex in the lift has degree 3. In the base graph, each vertex 
      has degree 3 (two cycle neighbors + one matching). In the lift, each vertex (v,0) 
      and (v,1) connects to neighbors based on signs. Since the base is 3-regular and 
      we're just assigning signs to the 2-lift, the lift is always 3-regular (each vertex 
      has the same number of edges as in the base).
    - Connected: we need to check if the lift is connected.
    
    So validity check: is the lift connected?
    """
    num_edges = len(edges)
    # Build signed adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = float(signs[i])
        A[u, v] = s
        A[v, u] = s
    
    # Check connectivity of the lift
    # The lift has 2n vertices: (v, 0) for v in 0..n-1 and (v, 1) for v in 0..n-1
    # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
    # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
    
    # Build adjacency list for the lift
    lift_adj = [[] for _ in range(2 * n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # +1: equal sides
            lift_adj[u].append(v)
            lift_adj[v].append(u)
            lift_adj[u + n].append(v + n)
            lift_adj[v + n].append(u + n)
        else:
            # -1: opposite sides
            lift_adj[u].append(v + n)
            lift_adj[v + n].append(u)
            lift_adj[u + n].append(v)
            lift_adj[v].append(u + n)
    
    # BFS/DFS to check connectivity
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    if not all(visited):
        return None, False  # Invalid: not connected
    
    # Compute spectral radius
    eigenvalues = np.linalg.eigvalsh(A)
    radius = max(abs(eigenvalues[0]), abs(eigenvalues[-1]))
    return radius, True

def _quick_radius_approx(n, edges, signs):
    """Quick approximation: use power iteration for spectral radius."""
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = float(signs[i])
        A[u, v] = s
        A[v, u] = s
    
    # Power iteration
    x = np.random.default_rng(42).uniform(-1, 1, size=n)
    x /= np.linalg.norm(x)
    for _ in range(50):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            break
        x /= norm
    return np.abs(x @ (A @ x))

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Deadline: 90 seconds wall clock, keep safety margin
    start_time = time.time()
    deadline = start_time + 80.0  # 80 seconds to be safe
    
    # Strategy: Use random search with the given budget, plus some structured search
    rng = np.random.default_rng(3)
    
    best_radius = float('inf')
    best_signs = None
    
    # First, try the seed approach: budget random sign vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        radius, valid = _evaluate_signs(n, edges, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs
    
    # Check time
    if time.time() > deadline:
        return best_signs if best_signs is not None else np.ones(num_edges, dtype=int)
    
    # Now do more extensive search with remaining time
    # For small m (4, 5, 7, 9), we can try more systematic approaches
    
    # Strategy 2: Random restarts with local search
    num_edges = len(edges)
    
    while time.time() < deadline - 5.0:  # Keep 5 seconds margin
        # Start from a random sign assignment
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        radius, valid = _evaluate_signs(n, edges, signs)
        
        if not valid:
            continue
        
        current_radius = radius
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try flipping each edge
            for i in range(num_edges):
                # Check time
                if time.time() > deadline - 5.0:
                    break
                
                signs[i] = -signs[i]
                new_radius, new_valid = _evaluate_signs(n, edges, signs)
                if new_valid and new_radius < current_radius:
                    current_radius = new_radius
                    best_radius = min(best_radius, current_radius)
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]  # revert
        
        # If we found a better solution, update
        if valid and current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
    
    # Return the best sign assignment found
    if best_signs is not None:
        return best_signs
    
    # Fallback: return all +1 signs (should be valid for prism graphs)
    return np.ones(num_edges, dtype=int)

# EVOLVE-BLOCK-END
