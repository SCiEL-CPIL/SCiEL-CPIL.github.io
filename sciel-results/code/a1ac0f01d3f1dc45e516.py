"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix."""
    n = graph['n']
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(graph['edges'], signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius (max absolute eigenvalue) of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_lift_connected(graph, signs):
    """Check if the 2-lift is connected.
    
    The 2-lift is disconnected iff the signed graph is frustration-free,
    i.e., there exists a vertex labeling c: V -> {+1, -1} such that
    for every edge (u,v) with sign s: c[u] * c[v] = s.
    Equivalently, the product of signs around every cycle is +1.
    """
    n = graph['n']
    # Build adjacency with signs
    adj = {}
    for i, (u, v) in enumerate(graph['edges']):
        s = signs[i]
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append((v, s))
        adj[v].append((u, s))
    
    # Try to find a consistent 2-coloring (frustration-free check)
    # c[v] in {+1, -1} such that c[u]*c[v] = s for edge (u,v) with sign s
    color = {}
    for start in range(n):
        if start in color:
            continue
        color[start] = 1
        queue = [start]
        while queue:
            u = queue.pop()
            for (v, s) in adj.get(u, []):
                expected_c = s * color[u]
                if v in color:
                    if color[v] != expected_c:
                        return True  # frustrated, lift is connected
                else:
                    color[v] = expected_c
                    queue.append(v)
    # If we got here without contradiction, the graph is frustration-free
    return False  # lift is disconnected

def _local_search(graph, initial_signs, max_iter=50000, deadline=None):
    """Local search: flip one edge at a time to minimize spectral radius while keeping lift connected."""
    signs = initial_signs.copy()
    A = _build_signed_adjacency(graph, signs)
    best_radius = _spectral_radius(A)
    
    if not _is_lift_connected(graph, signs):
        return None, float('inf')
    
    n_edges = len(graph['edges'])
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_edges):
            # Flip edge i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            # Check connectivity
            if not _is_lift_connected(graph, signs):
                signs[i] = old_sign
                continue
            
            # Compute spectral radius
            A = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(A)
            
            if radius < best_radius - 1e-12:
                best_radius = radius
                improved = True
            else:
                signs[i] = old_sign
        
        if not improved:
            break
    
    return signs, best_radius

def _smart_initial_signs(graph, m):
    """Generate a good initial frustrated signing."""
    n_edges = len(graph['edges'])
    signs = np.ones(n_edges, dtype=int)
    
    # Make exactly one edge negative to create frustration
    # The matching edges are the "cross" edges; making one negative creates a negative cycle
    # Edges are ordered: likely bottom cycle (m), top cycle (m), matching (m)
    # Let's make the first matching edge negative
    # Actually, let's try several starting points
    
    # Strategy: make one cycle have an odd number of negative edges
    # For a prism, the "natural" cycles are: bottom cycle, top cycle, and the "ladder" cycles
    # Let's make the bottom cycle frustrated by making one edge negative
    signs[0] = -1  # first edge of bottom cycle
    
    return signs

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Generate multiple initial frustrated signings and do local search on each
    best_signs = None
    best_radius = float('inf')
    
    # Try different initial frustrated patterns
    initial_patterns = []
    
    # Pattern 1: one negative edge in bottom cycle
    s1 = np.ones(n_edges, dtype=int)
    s1[0] = -1
    initial_patterns.append(s1)
    
    # Pattern 2: one negative edge in top cycle
    s2 = np.ones(n_edges, dtype=int)
    s2[m] = -1
    initial_patterns.append(s2)
    
    # Pattern 3: one negative matching edge
    s3 = np.ones(n_edges, dtype=int)
    s3[2*m] = -1
    initial_patterns.append(s3)
    
    # Pattern 4: two negative edges to create frustration differently
    s4 = np.ones(n_edges, dtype=int)
    s4[0] = -1
    s4[m] = -1
    initial_patterns.append(s4)
    
    # Pattern 5: all positive except one random
    rng = np.random.default_rng(42)
    for _ in range(5):
        s = np.ones(n_edges, dtype=int)
        idx = rng.integers(0, n_edges)
        s[idx] = -1
        initial_patterns.append(s)
    
    for init_signs in initial_patterns:
        if time.time() > deadline:
            break
        
        # Verify it's frustrated
        if not _is_lift_connected(graph, init_signs):
            continue
        
        signs, radius = _local_search(graph, init_signs, max_iter=10000, deadline=deadline)
        if signs is not None and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: return a simple frustrated pattern
        best_signs = np.ones(n_edges, dtype=int)
        best_signs[0] = -1
    
    return best_signs
# EVOLVE-BLOCK-END
