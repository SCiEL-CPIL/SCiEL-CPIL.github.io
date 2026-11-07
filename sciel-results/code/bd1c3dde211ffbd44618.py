"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n))
    for edge, s in zip(graph['edges'], signs):
        u, v = edge
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _exhaustive_search(graph, m):
    """Exhaustive search for small m."""
    n_edges = len(graph['edges'])
    best_signs = None
    best_radius = float('inf')
    
    # Iterate through all 2^n_edges combinations
    # For m=5, n_edges=15, 2^15 = 32768. This is fast.
    for mask in range(1 << n_edges):
        signs = np.array([(mask >> i) & 1 for i in range(n_edges)], dtype=int)
        signs = 2 * signs - 1 # Convert 0/1 to -1/1
        
        valid, radius = _is_valid_lift(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    return best_signs, best_radius

def _simulated_annealing(graph, m, time_limit=50):
    """Simulated Annealing for larger m."""
    n_edges = len(graph['edges'])
    rng = np.random.default_rng(123)
    
    # Initial state
    signs = rng.choice([-1, 1], size=n_edges)
    
    # Ensure validity
    valid, radius = _is_valid_lift(graph, signs)
    while not valid:
        # Flip a random edge
        idx = rng.integers(0, n_edges)
        signs[idx] *= -1
        valid, radius = _is_valid_lift(graph, signs)
        
    best_signs = signs.copy()
    best_radius = radius
    current_radius = radius
    
    # Annealing schedule
    start_time = time.time()
    T = 1.0
    T_min = 0.001
    alpha = 0.95
    
    while time.time() - start_time < time_limit:
        # Perturb: flip one random edge
        idx = rng.integers(0, n_edges)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        valid, new_radius = _is_valid_lift(graph, signs)
        
        if not valid:
            # Revert if invalid
            signs[idx] = old_sign
            T *= alpha
            continue
            
        delta = new_radius - current_radius
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_signs = signs.copy()
        else:
            # Reject
            signs[idx] = old_sign
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_signs, best_radius

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    # Determine strategy based on m
    if m <= 5:
        # Exhaustive search is feasible and optimal
        best_signs, best_radius = _exhaustive_search(graph, m)
        if best_signs is not None:
            return best_signs
    else:
        # Use Simulated Annealing
        best_signs, best_radius = _simulated_annealing(graph, m, time_limit=55)
        if best_signs is not None:
            return best_signs
            
    # Fallback if something went wrong (should not happen)
    from autoresearch_suite.science_tcs_expansion import lift_baseline
    return lift_baseline(graph, budget)
# EVOLVE-BLOCK-END
