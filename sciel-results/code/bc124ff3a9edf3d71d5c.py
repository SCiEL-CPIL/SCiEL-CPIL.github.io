"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_set = set()
    for i, (u, v) in enumerate(edges):
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Helper to get sign index for an edge
    def get_edge_idx(u, v):
        if (u, v) in edge_set:
            return edges.index((u, v))
        elif (v, u) in edge_set:
            return edges.index((v, u))
        return -1
    
    # Precompute edge indices for faster lookup
    edge_idx_map = {}
    for i, (u, v) in enumerate(edges):
        edge_idx_map[(u, v)] = i
        edge_idx_map[(v, u)] = i
    
    def get_edge_idx_fast(u, v):
        return edge_idx_map.get((u, v), -1)
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid(signs):
        """Check if the lift is valid using check_lift."""
        try:
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, float('inf')
    
    # Generate initial candidate signs
    candidates = []
    
    # Pattern 1: All -1
    candidates.append(np.ones(num_edges, dtype=int) * -1)
    
    # Pattern 2: Matching edges -1, cycle edges +1
    signs = np.ones(num_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            signs[i] = -1
    candidates.append(signs)
    
    # Pattern 3: Matching edges +1, cycle edges -1
    signs = np.ones(num_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            signs[i] = 1
        else:
            signs[i] = -1
    candidates.append(signs)
    
    # Pattern 4: Alternating cycle signs, matching -1
    for top_start in [-1, 1]:
        for bot_start in [-1, 1]:
            signs = np.ones(num_edges, dtype=int)
            for i, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    signs[i] = -1
                elif u < m and v < m:  # top cycle
                    # Determine position in cycle
                    if u == 0 and v == m-1 or v == 0 and u == m-1:
                        pos = m-1
                    else:
                        pos = min(u, v)
                    signs[i] = top_start * ((-1) ** pos)
                elif u >= m and v >= m:  # bottom cycle
                    if u == m and v == n-1 or v == m and u == n-1:
                        pos = m-1
                    else:
                        pos = min(u, v) - m
                    signs[i] = bot_start * ((-1) ** pos)
            candidates.append(signs)
    
    # Pattern 5: Random with seed
    rng = np.random.default_rng(3)
    for _ in range(20):
        signs = rng.choice([-1, 1], size=num_edges)
        candidates.append(signs)
    
    # Pattern 6: Structured patterns based on m
    # Try to make the signed graph have small spectral radius
    for pattern in range(10):
        signs = np.ones(num_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                # Matching edge
                signs[i] = -1 if (pattern % 2 == 0) else 1
            else:
                # Cycle edge
                if u < m and v < m:
                    pos = min(u, v)
                    signs[i] = -1 if (pos + pattern) % 2 == 0 else 1
                else:
                    pos = min(u, v) - m
                    signs[i] = -1 if (pos + pattern) % 2 == 0 else 1
        candidates.append(signs)
    
    # Evaluate all candidates
    best_signs = None
    best_radius = float('inf')
    
    for signs in candidates:
        valid, radius = is_valid(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If no valid candidate found, use all -1 as fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int) * -1
        best_radius, _ = is_valid(best_signs)
    
    # Local search: try flipping individual edges
    improved = True
    max_iterations = 5000
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(num_edges):
            trial = best_signs.copy()
            trial[i] *= -1
            valid, radius = is_valid(trial)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = trial
                improved = True
    
    # Second phase: try flipping pairs of edges
    improved = True
    iteration = 0
    while improved and iteration < 2000:
        improved = False
        iteration += 1
        for i in range(num_edges):
            for j in range(i+1, num_edges):
                trial = best_signs.copy()
                trial[i] *= -1
                trial[j] *= -1
                valid, radius = is_valid(trial)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = trial
                    improved = True
                    break
            if improved:
                break
    
    return best_signs
# EVOLVE-BLOCK-END
