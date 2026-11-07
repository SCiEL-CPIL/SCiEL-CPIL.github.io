"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _spectral_radius_signed(graph, signs):
    """Compute spectral radius of signed adjacency matrix."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = float(signs[i])
        A[u, v] = s
        A[v, u] = s
    try:
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    except:
        return np.inf

def _is_valid_lift(graph, signs):
    """Check if the lift is valid (connected, 3-regular)."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True
    except:
        return False

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Determine m
    m = n // 2
    
    # Build edge index structure for quick sign manipulation
    # Edge types: bottom cycle, top cycle, matching
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(i)
        elif u >= m and v >= m:
            top_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Strategy: Search over sign patterns
    # Key insight: For each cycle, the product of signs determines the "frustration"
    # We'll try both frustrated and unfrustrated configurations
    
    best_signs = None
    best_radius = np.inf
    
    # Initialize with a reasonable starting point
    # Try: all +1 for matching, alternate signs on cycles to minimize radius
    rng = np.random.default_rng(42)
    
    # First, try some structured patterns
    patterns_to_try = []
    
    # Pattern 1: All +1
    patterns_to_try.append(np.ones(num_edges, dtype=int))
    
    # Pattern 2: Bottom cycle all +1, top cycle all +1, matching alternating
    for match_pattern in [0, 1]:
        s = np.ones(num_edges, dtype=int)
        for j, idx in enumerate(matching_edges):
            if match_pattern == 1 and j % 2 == 1:
                s[idx] = -1
        patterns_to_try.append(s)
    
    # Pattern 3: Bottom cycle frustrated (one -1), top cycle unfrustrated
    for bottom_neg_idx in range(min(3, len(bottom_cycle_edges))):
        for top_neg_idx in range(min(3, len(top_cycle_edges))):
            s = np.ones(num_edges, dtype=int)
            if bottom_neg_idx < len(bottom_cycle_edges):
                s[bottom_cycle_edges[bottom_neg_idx]] = -1
            if top_neg_idx < len(top_cycle_edges):
                s[top_cycle_edges[top_neg_idx]] = -1
            patterns_to_try.append(s)
    
    # Evaluate initial patterns
    for s in patterns_to_try:
        if time.time() > deadline:
            break
        if _is_valid_lift(graph, s):
            r = _spectral_radius_signed(graph, s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
    
    # If no valid pattern found, use random search
    if best_signs is None:
        for _ in range(1000):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            if _is_valid_lift(graph, s):
                r = _spectral_radius_signed(graph, s)
                if r < best_radius:
                    best_radius = r
                    best_signs = s.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    # Local search: flip one edge at a time to improve
    current_signs = best_signs.copy()
    current_radius = _spectral_radius_signed(graph, current_signs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            # Try flipping edge i
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if _is_valid_lift(graph, trial):
                r = _spectral_radius_signed(graph, trial)
                if r < current_radius - 1e-10:
                    current_signs = trial
                    current_radius = r
                    improved = True
    
    # Multi-start local search if time permits
    if time.time() < deadline - 5.0 and best_radius > 0:
        for restart in range(20):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            if not _is_valid_lift(graph, s):
                continue
            r = _spectral_radius_signed(graph, s)
            
            # Local search from this start
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                trial = s.copy()
                trial[i] = -trial[i]
                if _is_valid_lift(graph, trial):
                    r_trial = _spectral_radius_signed(graph, trial)
                    if r_trial < r - 1e-10:
                        s = trial
                        r = r_trial
            
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
