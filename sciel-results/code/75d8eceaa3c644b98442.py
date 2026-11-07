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

def _make_frustrated_signs(m, edge_indices, bottom_cycle_edges, top_cycle_edges, matching_edges, frustration_pattern='both'):
    """
    Create signs with frustrated cycles.
    For even m: product around each cycle = -1 gives spectral radius contribution ~2cos(pi/m)
    For odd m: product around each cycle = -1 gives spectral radius contribution ~2cos(pi/(2m))
    
    frustration_pattern: 'both' (both cycles frustrated), 'bottom' (only bottom), 'top' (only top)
    """
    num_edges = len(edge_indices)
    signs = np.ones(num_edges, dtype=int)
    
    # Set matching signs to +1 (or we can vary these)
    for idx in matching_edges:
        signs[idx] = 1
    
    # For each cycle, set exactly one edge to -1 to make product = -1
    if frustration_pattern in ('both', 'bottom'):
        # Bottom cycle: set one edge to -1
        if len(bottom_cycle_edges) > 0:
            signs[bottom_cycle_edges[0]] = -1
    
    if frustration_pattern in ('both', 'top'):
        # Top cycle: set one edge to -1
        if len(top_cycle_edges) > 0:
            signs[top_cycle_edges[0]] = -1
    
    return signs

def _local_search(graph, signs, deadline, max_rounds=10):
    """Local search: flip one edge at a time to improve spectral radius."""
    current_signs = signs.copy()
    current_radius = _spectral_radius_signed(graph, current_signs)
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
                r = _spectral_radius_signed(graph, trial)
                if r < current_radius - 1e-12:
                    current_signs = trial
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
    
    # Build edge index structure
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
    
    best_signs = None
    best_radius = np.inf
    
    # Strategy: Generate frustrated cycle configurations and refine with local search
    frustration_patterns = ['both', 'bottom', 'top', 'neither']
    
    # For each frustration pattern, try different positions for the -1 sign
    candidates = []
    
    for pattern in frustration_patterns:
        if pattern == 'neither':
            # All +1
            s = np.ones(num_edges, dtype=int)
            candidates.append(s)
        elif pattern == 'both':
            # Try different positions for -1 in each cycle
            for bi in range(min(4, len(bottom_cycle_edges))):
                for ti in range(min(4, len(top_cycle_edges))):
                    s = np.ones(num_edges, dtype=int)
                    if bi < len(bottom_cycle_edges):
                        s[bottom_cycle_edges[bi]] = -1
                    if ti < len(top_cycle_edges):
                        s[top_cycle_edges[ti]] = -1
                    candidates.append(s)
        elif pattern == 'bottom':
            for bi in range(min(4, len(bottom_cycle_edges))):
                s = np.ones(num_edges, dtype=int)
                if bi < len(bottom_cycle_edges):
                    s[bottom_cycle_edges[bi]] = -1
                candidates.append(s)
        elif pattern == 'top':
            for ti in range(min(4, len(top_cycle_edges))):
                s = np.ones(num_edges, dtype=int)
                if ti < len(top_cycle_edges):
                    s[top_cycle_edges[ti]] = -1
                candidates.append(s)
    
    # Also add some random frustrated configurations
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        s = np.ones(num_edges, dtype=int)
        # Randomly set 1-2 edges in bottom cycle to -1
        if len(bottom_cycle_edges) > 0:
            num_neg_bottom = rng.integers(1, min(3, len(bottom_cycle_edges)) + 1)
            neg_indices = rng.choice(len(bottom_cycle_edges), size=num_neg_bottom, replace=False)
            for idx in neg_indices:
                s[bottom_cycle_edges[idx]] = -1
        # Randomly set 1-2 edges in top cycle to -1
        if len(top_cycle_edges) > 0:
            num_neg_top = rng.integers(1, min(3, len(top_cycle_edges)) + 1)
            neg_indices = rng.choice(len(top_cycle_edges), size=num_neg_top, replace=False)
            for idx in neg_indices:
                s[top_cycle_edges[idx]] = -1
        candidates.append(s)
    
    # Evaluate candidates and do local search on the best ones
    candidate_results = []
    for s in candidates:
        if time.time() > deadline:
            break
        if not _is_valid_lift(graph, s):
            continue
        r = _spectral_radius_signed(graph, s)
        candidate_results.append((r, s.copy()))
    
    # Sort by radius and do local search on top candidates
    candidate_results.sort(key=lambda x: x[0])
    
    for r, s in candidate_results[:10]:
        if time.time() > deadline - 5.0:
            break
        refined_signs, refined_radius = _local_search(graph, s, deadline, max_rounds=5)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()
    
    # If no valid candidate found, try random search
    if best_signs is None:
        for _ in range(200):
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
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        refined_signs, refined_radius = _local_search(graph, best_signs, deadline, max_rounds=10)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
