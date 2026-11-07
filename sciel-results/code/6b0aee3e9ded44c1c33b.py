"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, while ensuring the 2-lift is connected.
    
    Strategy: Use the "signed graph balancing" approach.
    For even m, enforce product of signs around every fundamental cycle to be -1.
    For odd m, this is impossible, so we generate candidates that satisfy
    top/bottom cycles = -1 and let ladder cycles be whatever, or vice versa,
    and pick the best valid one.
    """
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge indices for top, bottom, and matching
    top_idx = []
    bot_idx = []
    match_idx = []
    
    for i in range(m):
        u, v = i, (i + 1) % m
        for idx, (eu, ev) in enumerate(edges):
            if (eu == u and ev == v) or (eu == v and ev == u):
                top_idx.append(idx)
                break
        
        u, v = m + i, m + (i + 1) % m
        for idx, (eu, ev) in enumerate(edges):
            if (eu == u and ev == v) or (eu == v and ev == u):
                bot_idx.append(idx)
                break
        
        u, v = i, m + i
        for idx, (eu, ev) in enumerate(edges):
            if (eu == u and ev == v) or (eu == v and ev == u):
                match_idx.append(idx)
                break

    def build_signed_adjacency(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        A = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False

    candidates = []

    # Case 1: m is even. Exact solution exists where all fundamental cycles are -1.
    if m % 2 == 0:
        # Set match signs to +1
        s_match = np.ones(m, dtype=int)
        # Set bottom signs such that product is -1
        s_bot = np.ones(m, dtype=int)
        s_bot[0] = -1
        # Compute top signs: s_t[i] = - s_m[i+1] s_b[i] s_m[i]
        s_top = np.ones(m, dtype=int)
        for i in range(m):
            # Indices in s_match: i and (i+1)%m
            # Indices in s_bot: i
            s_top[i] = - (s_match[(i + 1) % m] * s_bot[i] * s_match[i])
        
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[top_idx[i]] = s_top[i]
            signs[bot_idx[i]] = s_bot[i]
            signs[match_idx[i]] = s_match[i]
        
        if is_valid(signs):
            candidates.append(signs)

    # Case 2: m is odd (or always add these as candidates).
    # Candidate A: Top -1, Bot -1, Match +1
    s_match = np.ones(m, dtype=int)
    s_bot = np.ones(m, dtype=int)
    s_bot[0] = -1
    s_top = np.ones(m, dtype=int)
    s_top[0] = -1
    
    signs_A = np.ones(num_edges, dtype=int)
    for i in range(m):
        signs_A[top_idx[i]] = s_top[i]
        signs_A[bot_idx[i]] = s_bot[i]
        signs_A[match_idx[i]] = s_match[i]
    if is_valid(signs_A):
        candidates.append(signs_A)

    # Candidate B: Top +1, Bot +1, Match -1 (or similar)
    # Let's try: Top +1, Bot -1, Match +1 (Same as A but Top +1)
    s_top = np.ones(m, dtype=int)
    s_bot = np.ones(m, dtype=int)
    s_bot[0] = -1
    s_match = np.ones(m, dtype=int)
    
    signs_B = np.ones(num_edges, dtype=int)
    for i in range(m):
        signs_B[top_idx[i]] = s_top[i]
        signs_B[bot_idx[i]] = s_bot[i]
        signs_B[match_idx[i]] = s_match[i]
    if is_valid(signs_B):
        candidates.append(signs_B)

    # Candidate C: Alternating signs on top
    s_top = np.ones(m, dtype=int)
    for i in range(1, m, 2):
        s_top[i] = -1
    s_bot = np.ones(m, dtype=int)
    s_match = np.ones(m, dtype=int)
    
    signs_C = np.ones(num_edges, dtype=int)
    for i in range(m):
        signs_C[top_idx[i]] = s_top[i]
        signs_C[bot_idx[i]] = s_bot[i]
        signs_C[match_idx[i]] = s_match[i]
    if is_valid(signs_C):
        candidates.append(signs_C)

    # Candidate D: Alternating signs on bottom
    s_top = np.ones(m, dtype=int)
    s_bot = np.ones(m, dtype=int)
    for i in range(1, m, 2):
        s_bot[i] = -1
    s_match = np.ones(m, dtype=int)
    
    signs_D = np.ones(num_edges, dtype=int)
    for i in range(m):
        signs_D[top_idx[i]] = s_top[i]
        signs_D[bot_idx[i]] = s_bot[i]
        signs_D[match_idx[i]] = s_match[i]
    if is_valid(signs_D):
        candidates.append(signs_D)

    if not candidates:
        # Fallback: random
        rng = np.random.default_rng(42)
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        if not is_valid(signs):
            # Try to fix
            for idx in range(num_edges):
                s_try = signs.copy()
                s_try[idx] = -s_try[idx]
                if is_valid(s_try):
                    signs = s_try
                    break
        candidates.append(signs)

    best_signs = None
    best_radius = float('inf')
    for s in candidates:
        r = spectral_radius(s)
        if r < best_radius:
            best_radius = r
            best_signs = s

    return best_signs
# EVOLVE-BLOCK-END
