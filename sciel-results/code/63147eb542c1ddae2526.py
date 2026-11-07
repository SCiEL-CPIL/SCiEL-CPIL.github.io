"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

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

def _generate_candidates(graph, m):
    """Generate a set of structured candidate sign vectors."""
    n_edges = len(graph['edges'])
    candidates = []
    
    # Identify edge indices
    top_edges = []
    bot_edges = []
    match_edges = []
    for i, (u, v) in enumerate(graph['edges']):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bot_edges.append(i)
        else:
            match_edges.append(i)
            
    # Helper to create a sign vector
    def make_signs(top_signs, bot_signs, match_signs):
        signs = np.ones(n_edges, dtype=int)
        for i, idx in enumerate(top_edges):
            signs[idx] = top_signs[i]
        for i, idx in enumerate(bot_edges):
            signs[idx] = bot_signs[i]
        for i, idx in enumerate(match_edges):
            signs[idx] = match_signs[i]
        return signs

    # 1. All +1
    candidates.append(make_signs(np.ones(m), np.ones(m), np.ones(m)))
    
    # 2. All -1
    candidates.append(make_signs(-np.ones(m), -np.ones(m), -np.ones(m)))
    
    # 3. Matching -1, Cycles +1
    candidates.append(make_signs(np.ones(m), np.ones(m), -np.ones(m)))
    
    # 4. Matching +1, Top -1, Bottom +1
    candidates.append(make_signs(-np.ones(m), np.ones(m), np.ones(m)))
    
    # 5. Matching +1, Top +1, Bottom -1
    candidates.append(make_signs(np.ones(m), -np.ones(m), np.ones(m)))
    
    # 6. Alternating on cycles (if m is even)
    if m % 2 == 0:
        alt = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
        candidates.append(make_signs(alt, alt, np.ones(m)))
        candidates.append(make_signs(alt, -alt, np.ones(m)))
        candidates.append(make_signs(alt, alt, -np.ones(m)))
        
    # 7. "One frustrated edge" patterns
    # Top: one -1, rest +1. Bottom: one -1, rest +1. Match: +1
    for i in range(m):
        t = np.ones(m); t[i] = -1
        b = np.ones(m); b[(i+1)%m] = -1 # Offset
        candidates.append(make_signs(t, b, np.ones(m)))
        
        t = np.ones(m); t[i] = -1
        b = np.ones(m); b[i] = -1 # Aligned
        candidates.append(make_signs(t, b, np.ones(m)))
        
        # Matching -1
        candidates.append(make_signs(t, b, -np.ones(m)))

    # 8. Random samples
    rng = np.random.default_rng(3)
    for _ in range(20):
        candidates.append(rng.choice([-1, 1], size=n_edges))
        
    return candidates

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    candidates = _generate_candidates(graph, m)
    
    best_signs = None
    best_radius = float('inf')
    
    for signs in candidates:
        valid, radius = _is_valid_lift(graph, signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: use the seed approach
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
        
    return best_signs
# EVOLVE-BLOCK-END
