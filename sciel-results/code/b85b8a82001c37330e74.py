"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _signed_adj_matrix(n, edges, signs):
    """Build the signed adjacency matrix."""
    A = np.zeros((n, n))
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _is_valid_lift(n, edges, signs):
    """Check if the lift is connected and 3-regular."""
    # The lift has 2n vertices
    lift_adj = {}
    for i in range(2 * n):
        lift_adj[i] = []
    
    for (u, v), s in zip(edges, signs):
        if s == 1:
            lift_adj[u].append(v)
            lift_adj[n + u].append(n + v)
            lift_adj[v].append(u)
            lift_adj[n + v].append(n + u)
        else:
            lift_adj[u].append(n + v)
            lift_adj[n + u].append(v)
            lift_adj[v].append(n + u)
            lift_adj[n + v].append(u)
    
    # Check 3-regular
    for i in range(2 * n):
        if len(lift_adj[i]) != 3:
            return False
    
    # Check connectivity using BFS
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == 2 * n

def _spectral_radius(n, edges, signs):
    """Compute the spectral radius of the signed adjacency matrix."""
    A = _signed_adj_matrix(n, edges, signs)
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def sign_edges(graph, budget):
    """Find signs for edges to minimize the spectral radius of the signed adjacency matrix."""
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Precompute edge indices by type
    cycle1_edges = []
    cycle2_edges = []
    matching_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_edges.append(idx)
        elif u >= m and v >= m:
            cycle2_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Phase 1: Phase-synchronization initialization
    # For each cycle, the product of signs around the cycle determines
    # whether the lift connects the two sides or not.
    # We want to find sign assignments that minimize spectral radius.
    
    # Generate diverse initial candidates using phase approach
    candidates = []
    
    # Candidate 1: All +1
    signs = np.ones(num_edges, dtype=int)
    candidates.append(signs)
    
    # Candidate 2: All -1
    signs = np.ones(num_edges, dtype=int) * -1
    candidates.append(signs)
    
    # Candidates 3-8: Various phase patterns on cycles
    for phase_shift in range(6):
        signs = np.ones(num_edges, dtype=int)
        # Set cycle1 edges with alternating pattern based on phase
        for i, idx in enumerate(cycle1_edges):
            if (i + phase_shift) % 2 == 1:
                signs[idx] = -1
        candidates.append(signs.copy())
    
    # More random candidates
    for _ in range(10):
        signs = rng.choice([-1, 1], size=num_edges)
        candidates.append(signs)
    
    # Evaluate all candidates
    for signs in candidates:
        if time.time() > deadline - 10.0:
            break
        if _is_valid_lift(n, edges, signs):
            radius = _spectral_radius(n, edges, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    # Phase 2: Local search with 2-opt moves
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = _spectral_radius(n, edges, best_signs)
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # 2-opt local search: try flipping pairs of edges
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Try random 2-opt moves
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            
            i, j = rng.choice(num_edges, size=2, replace=False)
            if i == j:
                continue
            
            trial = current_signs.copy()
            trial[i] *= -1
            trial[j] *= -1
            
            if _is_valid_lift(n, edges, trial):
                trial_radius = _spectral_radius(n, edges, trial)
                if trial_radius < current_radius:
                    current_signs = trial
                    current_radius = trial_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
        
        # Also try single flips
        i = rng.integers(0, num_edges)
        trial = current_signs.copy()
        trial[i] *= -1
        
        if _is_valid_lift(n, edges, trial):
            trial_radius = _spectral_radius(n, edges, trial)
            if trial_radius < current_radius:
                current_signs = trial
                current_radius = trial_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
    
    # Phase 3: Systematic polish - try all single flips
    if time.time() < deadline - 3.0:
        for i in range(num_edges):
            if time.time() > deadline - 3.0:
                break
            trial = current_signs.copy()
            trial[i] *= -1
            if _is_valid_lift(n, edges, trial):
                trial_radius = _spectral_radius(n, edges, trial)
                if trial_radius < current_radius:
                    current_signs = trial
                    current_radius = trial_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
    
    return best_signs
# EVOLVE-BLOCK-END
