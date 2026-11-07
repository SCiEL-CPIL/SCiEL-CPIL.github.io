"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _cycle_eigenvalues(m, signs_cycle):
    """
    Compute eigenvalues of a signed m-cycle.
    signs_cycle: array of length m with +1/-1 for each edge.
    The cycle adjacency has A[i, (i+1)%m] = s_i, A[(i+1)%m, i] = s_i.
    
    Eigenvalues: 2*cos(2*pi*k/m + theta) where theta depends on product of signs.
    Product P = prod(s_i). If P = +1, theta = 0. If P = -1, theta = pi/m.
    Actually for signed cycle: eigenvalues are 2*cos(2*pi*k/m) if all +1.
    More generally, the eigenvalues of a signed cycle with product P:
    - If P = +1: 2*cos(2*pi*k/m) for k=0,...,m-1
    - If P = -1: 2*cos((2k+1)*pi/m) for k=0,...,m-1
    """
    P = int(np.prod(signs_cycle))
    if P == 1:
        k = np.arange(m)
        return 2.0 * np.cos(2.0 * np.pi * k / m)
    else:
        k = np.arange(m)
        return 2.0 * np.cos((2.0 * k + 1.0) * np.pi / m)

def _spectral_radius_analytical(m, bottom_signs, top_signs, matching_signs):
    """
    Compute spectral radius of signed adjacency matrix of prism graph analytically.
    
    The signed adjacency matrix A has the form:
    A = [[B, M], [M, C]]
    where B is the bottom signed cycle (m x m), C is the top signed cycle (m x m),
    and M is the matching diagonal matrix with matching_signs on the diagonal.
    
    Both B and C are circulant matrices (up to sign patterns), so they can be 
    diagonalized by the DFT. M is diagonal.
    
    Actually, B and C are not necessarily circulant if signs vary. But they are 
    signed cycles, which are circulant up to a gauge transformation.
    
    Let me use the fact that for a signed cycle, the eigenvalues are known.
    The key is that B and C can be simultaneously block-diagonalized with M.
    
    Actually, let's use the following approach:
    The matrix A = [[B, M], [M, C]] where B, C are m x m signed cycle matrices
    and M is m x m diagonal.
    
    Since B is a signed cycle, it's a circulant matrix (each row is a shift of the previous,
    up to the sign pattern). Actually no - a signed cycle where edge (i, i+1) has sign s_i
    gives B[i, i+1] = s_i and B[i+1, i] = s_i (mod m). This is NOT circulant in general
    unless all s_i are the same.
    
    However, we can use a gauge transformation. Let D be diagonal with D[i] = product of 
    signs from vertex 0 to vertex i. Then D^{-1} B D is the standard cycle (all +1 signs).
    So B = D B_std D^{-1} where B_std has all +1 signs.
    
    Similarly for C: C = E C_std E^{-1}.
    
    Then A = [[D B_std D^{-1}, M], [M, E C_std E^{-1}]]
    
    This doesn't immediately help with block diagonalization.
    
    Alternative approach: Just build the matrix and compute eigvalsh. For m up to 9, 
    the matrix is 18x18 at most, which is very fast.
    """
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    
    # Bottom cycle: vertices 0 to m-1
    for i in range(m):
        j = (i + 1) % m
        s = float(bottom_signs[i])
        A[i, j] = s
        A[j, i] = s
    
    # Top cycle: vertices m to 2m-1
    for i in range(m):
        j = m + (i + 1) % m
        s = float(top_signs[i])
        A[m + i, j] = s
        A[j, m + i] = s
    
    # Matching: vertex i to vertex i+m
    for i in range(m):
        s = float(matching_signs[i])
        A[i, m + i] = s
        A[m + i, i] = s
    
    eigvals = np.linalg.eigvalsh(A)
    return max(abs(eigvals[0]), abs(eigvals[-1]))

def _is_valid_lift(graph, signs):
    """Check if the lift is valid (connected, 3-regular)."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True
    except:
        return False

def _extract_cycle_signs(graph, m):
    """Extract bottom cycle signs, top cycle signs, and matching signs from edge list."""
    edges = graph['edges']
    bottom_signs = np.ones(m, dtype=int)
    top_signs = np.ones(m, dtype=int)
    matching_signs = np.ones(m, dtype=int)
    
    # Map edges to their types
    bottom_edge_map = {}  # edge index -> position in cycle
    top_edge_map = {}
    matching_edge_map = {}
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Bottom cycle edge
            # Determine position: edge from i to (i+1)%m
            a, b = min(u, v), max(u, v)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b  # edge from m-1 to 0
            bottom_edge_map[i] = pos
        elif u >= m and v >= m:
            # Top cycle edge
            a, b = min(u - m, v - m), max(u - m, v - m)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b
            top_edge_map[i] = pos
        else:
            # Matching edge
            pos = min(u, v) % m
            matching_edge_map[i] = pos
    
    return bottom_edge_map, top_edge_map, matching_edge_map

def _signs_to_components(signs, bottom_edge_map, top_edge_map, matching_edge_map, m):
    """Convert full signs array to (bottom_signs, top_signs, matching_signs)."""
    bottom_signs = np.ones(m, dtype=int)
    top_signs = np.ones(m, dtype=int)
    matching_signs = np.ones(m, dtype=int)
    
    for edge_idx, pos in bottom_edge_map.items():
        bottom_signs[pos] = signs[edge_idx]
    for edge_idx, pos in top_edge_map.items():
        top_signs[pos] = signs[edge_idx]
    for edge_idx, pos in matching_edge_map.items():
        matching_signs[pos] = signs[edge_idx]
    
    return bottom_signs, top_signs, matching_signs

def _local_search(graph, signs, deadline, max_rounds=15):
    """Local search: flip one edge at a time to improve spectral radius."""
    m = graph['n'] // 2
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    current_signs = signs.copy()
    b_s, t_s, m_s = _signs_to_components(current_signs, bottom_edge_map, top_edge_map, matching_edge_map, m)
    current_radius = _spectral_radius_analytical(m, b_s, t_s, m_s)
    num_edges = len(current_signs)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if _is_valid_lift(graph, trial):
                b_t, t_t, m_t = _signs_to_components(trial, bottom_edge_map, top_edge_map, matching_edge_map, m)
                r = _spectral_radius_analytical(m, b_t, t_t, m_t)
                if r < current_radius - 1e-12:
                    current_signs = trial
                    b_s, t_s, m_s = b_t, t_t, m_t
                    current_radius = r
                    improved = True
        if not improved:
            break
    
    return current_signs, current_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    best_signs = None
    best_radius = np.inf
    
    # Strategy: Systematically explore sign patterns
    # Key insight: the spectral radius depends on the product of signs around each cycle
    # and the matching signs. Let's enumerate all possible products.
    
    # For a signed cycle of length m:
    # - If product = +1: eigenvalues are 2*cos(2*pi*k/m), max |eig| = 2
    # - If product = -1: eigenvalues are 2*cos((2k+1)*pi/m), max |eig| = 2*cos(pi/m) for even m,
    #   or 2*cos(pi/(2m)) for odd m... let me compute properly.
    
    # For product = -1, eigenvalues are 2*cos((2k+1)*pi/m) for k=0,...,m-1
    # The max absolute value is 2*cos(pi/m) if m is even (k=0 gives 2cos(pi/m)),
    # and for odd m, the max is 2*cos(pi/m) as well (since (2k+1)*pi/m ranges over odd multiples).
    
    # Actually for m=4: product=-1 gives eigenvalues 2cos(pi/4), 2cos(3pi/4), 2cos(5pi/4), 2cos(7pi/4)
    # = sqrt(2), -sqrt(2), -sqrt(2), sqrt(2). Max abs = sqrt(2) ≈ 1.414
    # For m=5: 2cos(pi/5), 2cos(3pi/5), 2cos(5pi/5), 2cos(7pi/5), 2cos(9pi/5)
    # = 2cos(36°), 2cos(108°), 2cos(180°), 2cos(252°), 2cos(324°)
    # = 1.618, -0.618, -2, -0.618, 1.618. Max abs = 2.
    # Hmm, so for odd m, product=-1 can still give eigenvalue -2.
    
    # For m=7: 2cos(pi/7), 2cos(3pi/7), 2cos(5pi/7), 2cos(7pi/7), 2cos(9pi/7), 2cos(11pi/7), 2cos(13pi/7)
    # 2cos(pi/7)≈1.802, 2cos(3pi/7)≈0.445, 2cos(5pi/7)≈-1.247, 2cos(pi)=-2, ...
    # So max abs = 2 again for odd m.
    
    # So frustration (product=-1) only helps for even m.
    # For m=4: frustrated cycle has max |eig| = sqrt(2) ≈ 1.414 instead of 2.
    # For m=5,7,9: frustrated cycle still has max |eig| = 2.
    
    # The full spectral radius of the prism depends on all three components.
    # Let's do a systematic search over all 2^(3m) possible sign assignments.
    # For m=4: 2^12 = 4096 possibilities. Very fast.
    # For m=5: 2^15 = 32768. Still fast.
    # For m=7: 2^21 = 2M. Might be slow but let's try with the analytical formula.
    # For m=9: 2^27 = 134M. Too slow for full enumeration.
    
    # Strategy:
    # For m <= 5: enumerate all possibilities
    # For m > 5: use smarter search
    
    if m <= 5:
        # Full enumeration
        all_combos = np.array(list(product([1, -1], repeat=3 * m)))
        for combo in all_combos:
            if time.time() > deadline:
                break
            b_s = combo[:m].astype(int)
            t_s = combo[m:2*m].astype(int)
            m_s = combo[2*m:3*m].astype(int)
            r = _spectral_radius_analytical(m, b_s, t_s, m_s)
            if r < best_radius:
                # Build full signs array
                signs = np.ones(num_edges, dtype=int)
                for edge_idx, pos in bottom_edge_map.items():
                    signs[edge_idx] = b_s[pos]
                for edge_idx, pos in top_edge_map.items():
                    signs[edge_idx] = t_s[pos]
                for edge_idx, pos in matching_edge_map.items():
                    signs[edge_idx] = m_s[pos]
                
                if _is_valid_lift(graph, signs):
                    best_radius = r
                    best_signs = signs.copy()
    else:
        # Smarter search for m > 5
        # Key observation: for odd m, the cycle eigenvalue spectrum is the same whether
        # product is +1 or -1 (both include eigenvalue -2). So frustration doesn't help
        # for odd m cycles. But the matching signs and relative phases matter.
        
        # Let's try a structured search:
        # 1. Try all combinations of matching signs (2^m possibilities)
        # 2. For each, try a few cycle sign patterns
        
        candidates = []
        
        # Pattern 1: All +1
        s = np.ones(num_edges, dtype=int)
        candidates.append(s)
        
        # Pattern 2: Frustrate both cycles (one -1 each)
        for bi in range(m):
            for ti in range(m):
                s = np.ones(num_edges, dtype=int)
                s[list(bottom_edge_map.keys())[bi]] = -1
                s[list(top_edge_map.keys())[ti]] = -1
                candidates.append(s)
        
        # Pattern 3: Frustrate only bottom
        for bi in range(m):
            s = np.ones(num_edges, dtype=int)
            s[list(bottom_edge_map.keys())[bi]] = -1
            candidates.append(s)
        
        # Pattern 4: Frustrate only top
        for ti in range(m):
            s = np.ones(num_edges, dtype=int)
            s[list(top_edge_map.keys())[ti]] = -1
            candidates.append(s)
        
        # Pattern 5: Vary matching signs
        for mask in range(min(100, 2 ** m)):
            s = np.ones(num_edges, dtype=int)
            for i in range(m):
                if mask & (1 << i):
                    matching_edge = list(matching_edge_map.keys())[i]
                    s[matching_edge] = -1
            candidates.append(s)
        
        # Pattern 6: Random samples
        rng = np.random.default_rng(42)
        for _ in range(500):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            candidates.append(s)
        
        # Evaluate candidates
        candidate_results = []
        for s in candidates:
            if time.time() > deadline:
                break
            if not _is_valid_lift(graph, s):
                continue
            b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
            r = _spectral_radius_analytical(m, b_s, t_s, m_s)
            candidate_results.append((r, s.copy()))
        
        candidate_results.sort(key=lambda x: x[0])
        
        for r, s in candidate_results[:20]:
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
        
        # Local search on top candidates
        for r, s in candidate_results[:10]:
            if time.time() > deadline - 5.0:
                break
            refined_signs, refined_radius = _local_search(graph, s, deadline, max_rounds=10)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_signs = refined_signs.copy()
    
    # If no valid candidate found, try random search
    if best_signs is None:
        rng = np.random.default_rng(0)
        for _ in range(500):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            if _is_valid_lift(graph, s):
                b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
                r = _spectral_radius_analytical(m, b_s, t_s, m_s)
                if r < best_radius:
                    best_radius = r
                    best_signs = s.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        refined_signs, refined_radius = _local_search(graph, best_signs, deadline, max_rounds=15)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
