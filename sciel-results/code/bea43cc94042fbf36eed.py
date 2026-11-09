"""Optimized sign assignment for spectral graph lifts."""
import time
import numpy as np

def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = float(s)
        A[v, u] = float(s)
    return A

def _spectral_radius(A):
    """Compute spectral radius (max abs eigenvalue)."""
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')

def _check_lift_valid(graph, signs):
    """Check if the lift is valid (connected and 3-regular)."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, float(radius)
    except Exception:
        return False, float('inf')

def _initial_signs(graph):
    """Generate initial sign pattern: alternate on cycles, +1 on matching."""
    m = graph['n'] // 2
    edges = graph['edges']
    signs = np.ones(len(edges), dtype=np.int8)
    
    # Identify edge types and assign alternating signs on cycles
    # Edges in first cycle: (i, i+1) for i in 0..m-2, and (m-1, 0)
    # Edges in second cycle: (m+i, m+i+1) for i in 0..m-2, and (m+m-1, m)
    # Matching: (i, i+m)
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # First cycle
            # Alternate signs based on edge position
            # Edge from i to i+1 gets sign (-1)^i
            i = u
            if v == (u + 1) % m:
                signs[idx] = 1 if i % 2 == 0 else -1
        elif u >= m and v >= m:
            # Second cycle
            i = u - m
            if v == m + (i + 1) % m:
                signs[idx] = 1 if i % 2 == 0 else -1
        else:
            # Matching edge
            signs[idx] = 1
    
    return signs

def _local_search(graph, signs, max_iter=5000, time_limit=None, start_time=None):
    """Local search: flip signs to minimize spectral radius."""
    n_edges = len(graph['edges'])
    best_signs = signs.copy()
    best_radius = _spectral_radius(_build_signed_adj(graph, best_signs))
    
    # Check validity of initial
    valid, r = _check_lift_valid(graph, best_signs)
    if not valid:
        best_radius = float('inf')
    
    edges = graph['edges']
    
    for iteration in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n_edges):
            # Flip sign i
            new_signs = best_signs.copy()
            new_signs[i] = -new_signs[i]
            
            # Quick check: compute spectral radius
            A = _build_signed_adj(graph, new_signs)
            radius = _spectral_radius(A)
            
            if radius < best_radius - 1e-10:
                # Check validity
                valid, r = _check_lift_valid(graph, new_signs)
                if valid:
                    best_signs = new_signs
                    best_radius = radius
                    improved = True
                    break  # Restart scan after improvement
        
        if not improved:
            break
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    """Main entry point: assign signs to minimize spectral radius."""
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    best_signs = None
    best_radius = float('inf')
    
    # Multiple restarts with different initial patterns
    patterns = []
    
    # Pattern 1: alternating on both cycles, +1 on matching
    patterns.append('alt_alt_plus')
    # Pattern 2: alternating on first, alternating on second, -1 on matching
    patterns.append('alt_alt_minus')
    # Pattern 3: all +1
    patterns.append('all_plus')
    # Pattern 4: alternating on first, constant on second
    patterns.append('alt_const')
    # Pattern 5: constant on first, alternating on second
    patterns.append('const_alt')
    
    for pattern in patterns:
        if time.time() - start_time > time_limit:
            break
        
        # Generate initial signs based on pattern
        signs = np.ones(n_edges, dtype=np.int8)
        
        for idx, (u, v) in enumerate(graph['edges']):
            if u < m and v < m:
                # First cycle edge
                i = u
                if pattern in ['alt_alt_plus', 'alt_alt_minus', 'alt_const']:
                    signs[idx] = 1 if i % 2 == 0 else -1
                else:
                    signs[idx] = 1
            elif u >= m and v >= m:
                # Second cycle edge
                i = u - m
                if pattern in ['alt_alt_plus', 'alt_alt_minus', 'const_alt']:
                    signs[idx] = 1 if i % 2 == 0 else -1
                else:
                    signs[idx] = 1
            else:
                # Matching edge
                if pattern == 'alt_alt_minus':
                    signs[idx] = -1
                else:
                    signs[idx] = 1
        
        # Run local search
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining <= 0:
            break
        
        result_signs, result_radius = _local_search(
            graph, signs, 
            max_iter=2000,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if result_radius < best_radius:
            best_radius = result_radius
            best_signs = result_signs
    
    # If no valid solution found, return all +1
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs
