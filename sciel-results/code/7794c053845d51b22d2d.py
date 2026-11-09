"""Local-search + multi-seed approach for spectral graph lift sign assignment."""
import time
import numpy as np


def _build_signed_adj(n, edges, signs):
    A = np.zeros((n, n), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(A):
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))


def _is_valid_lift(graph, signs):
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, float(radius)
    except Exception:
        return False, float('inf')


def _flip_edge(signs, i):
    signs[i] = -signs[i]


def _local_search(graph, signs, A, best_radius, deadline, max_iter=20000):
    """Hill-climbing: flip single edges (and occasionally 2-flips) to reduce spectral radius."""
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    signs = signs.copy()
    current_radius = best_radius
    
    # Precompute edge endpoints for fast adjacency update
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try single-edge flips
        best_flip = -1
        best_new_radius = current_radius
        
        # Shuffle edge order to avoid bias
        order = np.random.default_rng(it * 7919 + 13).permutation(num_edges)
        
        for ei in order:
            if time.time() > deadline:
                break
            u, v = edge_u[ei], edge_v[ei]
            old_s = signs[ei]
            # Flip
            signs[ei] = -old_s
            A[u, v] = -old_s
            A[v, u] = -old_s
            
            # Check connectivity quickly: the lift is disconnected iff the product of
            # signs along the "top cycle" equals the product along the "bottom cycle"
            # in a way that decouples. Actually, for a prism graph, the 2-lift is
            # connected iff the product of signs around the top m-cycle times the
            # product around the bottom m-cycle... Let me just use check_lift.
            valid, radius = _is_valid_lift(graph, signs)
            
            if valid and radius < best_new_radius:
                best_new_radius = radius
                best_flip = ei
                best_u, best_v, best_old_s = u, v, old_s
            else:
                # Revert
                signs[ei] = old_s
                A[u, v] = old_s
                A[v, u] = old_s
        
        if best_flip >= 0:
            # Apply the best flip
            signs[best_flip] = -best_old_s
            A[best_u, best_v] = -best_old_s
            A[best_v, best_u] = -best_old_s
            current_radius = best_new_radius
        else:
            # Try 2-flip moves to escape local minima
            found_2flip = False
            for _ in range(min(num_edges, 40)):
                if time.time() > deadline:
                    break
                i1 = np.random.default_rng(it * 31 + _ * 17).integers(0, num_edges)
                i2 = np.random.default_rng(it * 31 + _ * 17 + 1).integers(0, num_edges)
                if i1 == i2:
                    continue
                u1, v1 = edge_u[i1], edge_v[i1]
                u2, v2 = edge_u[i2], edge_v[i2]
                s1, s2 = signs[i1], signs[i2]
                
                signs[i1] = -s1
                signs[i2] = -s2
                A[u1, v1] = -s1
                A[v1, u1] = -s1
                A[u2, v2] = -s2
                A[v2, u2] = -s2
                
                valid, radius = _is_valid_lift(graph, signs)
                
                if valid and radius < current_radius:
                    current_radius = radius
                    found_2flip = True
                    break
                else:
                    # Revert
                    signs[i1] = s1
                    signs[i2] = s2
                    A[u1, v1] = s1
                    A[v1, u1] = s1
                    A[u2, v2] = s2
                    A[v2, u2] = s2
            
            if not found_2flip:
                # No improvement found; break
                break
    
    return signs, current_radius


def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(3)
    
    candidates = []
    
    # 1. Random candidates
    for _ in range(50):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.float64)
        candidates.append(signs)
    
    # 2. Structured patterns: alternate signs on cycles
    # Top cycle edges: (0,1),(1,2),...,(m-1,0)
    # Bottom cycle edges: (m,m+1),...,(2m-1,m)
    # Matching edges: (i, i+m) for i in 0..m-1
    
    # Identify edge types
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Pattern: all +1
    signs = np.ones(num_edges, dtype=np.float64)
    candidates.append(signs.copy())
    
    # Pattern: alternating on top cycle, all +1 elsewhere
    signs = np.ones(num_edges, dtype=np.float64)
    for i, ei in enumerate(top_cycle_edges):
        signs[ei] = 1.0 if i % 2 == 0 else -1.0
    candidates.append(signs.copy())
    
    # Pattern: alternating on bottom cycle
    signs = np.ones(num_edges, dtype=np.float64)
    for i, ei in enumerate(bottom_cycle_edges):
        signs[ei] = 1.0 if i % 2 == 0 else -1.0
    candidates.append(signs.copy())
    
    # Pattern: alternating on both cycles
    signs = np.ones(num_edges, dtype=np.float64)
    for i, ei in enumerate(top_cycle_edges):
        signs[ei] = 1.0 if i % 2 == 0 else -1.0
    for i, ei in enumerate(bottom_cycle_edges):
        signs[ei] = 1.0 if i % 2 == 0 else -1.0
    candidates.append(signs.copy())
    
    # Pattern: matching edges all -1
    signs = np.ones(num_edges, dtype=np.float64)
    for ei in matching_edges:
        signs[ei] = -1.0
    candidates.append(signs.copy())
    
    # Pattern: matching edges alternating
    signs = np.ones(num_edges, dtype=np.float64)
    for i, ei in enumerate(matching_edges):
        signs[ei] = 1.0 if i % 2 == 0 else -1.0
    candidates.append(signs.copy())
    
    # Evaluate all candidates, find best valid
    best_signs = None
    best_radius = float('inf')
    
    for signs in candidates:
        if time.time() > deadline:
            break
        A = _build_signed_adj(n, edges, signs)
        valid, radius = _is_valid_lift(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 should always be valid (connected, 3-regular)
        best_signs = np.ones(num_edges, dtype=np.float64)
        best_radius = _spectral_radius(_build_signed_adj(n, edges, best_signs))
    
    # Local search from best candidate
    A = _build_signed_adj(n, edges, best_signs)
    best_signs, best_radius = _local_search(graph, best_signs, A, best_radius, deadline)
    
    # Additional local search from a few other good candidates
    for signs in candidates[:20]:
        if time.time() > deadline:
            break
        A = _build_signed_adj(n, edges, signs)
        valid, radius = _is_valid_lift(graph, signs)
        if valid and radius < best_radius:
            best_signs, best_radius = _local_search(graph, signs, A, radius, deadline)
    
    return best_signs.astype(np.int8)
