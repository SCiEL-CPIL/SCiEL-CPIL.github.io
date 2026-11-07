"""Spectral graph lifts: minimize spectral radius of signed adjacency matrix."""
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build adjacency structure
    # For prism graph: two m-cycles joined by matching
    # Vertices 0..m-1 form one cycle, m..2m-1 form the other
    # Matching edges: (i, i+m) for i in 0..m-1

    # Build edge index mapping
    edge_to_uv = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_uv[idx] = (u, v)

    # Build adjacency list with edge indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def compute_signed_adj(signs):
        """Compute signed adjacency matrix from signs."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        return A

    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = compute_signed_adj(signs)
        # For symmetric matrix, eigenvalues are real
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        """Check if the 2-lift is connected."""
        # Build lift graph: 2n vertices
        # Vertices (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u, v) with sign +1: joins (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u, v) with sign -1: joins (u,0)-(v,1) and (u,1)-(v,0)
        
        lift_n = 2 * n
        lift_adj = np.zeros((lift_n, lift_n), dtype=int)
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Equal sides
                lift_adj[u, v] = 1
                lift_adj[v, u] = 1
                lift_adj[u + n, v + n] = 1
                lift_adj[v + n, u + n] = 1
            else:
                # Opposite sides
                lift_adj[u, v + n] = 1
                lift_adj[v + n, u] = 1
                lift_adj[u + n, v] = 1
                lift_adj[v, u + n] = 1
        
        # Check connectivity
        n_components, _ = connected_components(lift_adj, directed=False)
        return n_components == 1

    def evaluate(signs):
        """Evaluate signs: return (spectral_radius, is_valid)."""
        if not is_connected_lift(signs):
            return np.inf, False
        sr = spectral_radius(signs)
        return sr, True

    # Strategy: Start with random signs, then do local search (flip edges)
    # to minimize spectral radius while maintaining connectivity.

    best_signs = None
    best_radius = np.inf

    # Use multiple random restarts
    rng = np.random.default_rng(3)

    # Try structured initializations first
    # For prism graphs, we can try specific patterns
    
    # Pattern 1: All +1
    signs = np.ones(num_edges, dtype=int)
    sr, valid = evaluate(signs)
    if valid and sr < best_radius:
        best_radius = sr
        best_signs = signs.copy()

    # Pattern 2: Alternating signs on cycles
    for pattern_seed in range(10):
        signs = rng.choice([-1, 1], size=num_edges)
        sr, valid = evaluate(signs)
        if valid and sr < best_radius:
            best_radius = sr
            best_signs = signs.copy()

    # Local search: flip single edges
    def local_search(signs, max_time):
        """Perform local search by flipping single edges."""
        current_signs = signs.copy()
        current_sr, valid = evaluate(current_signs)
        if not valid:
            return current_signs, np.inf
        
        best_local_sr = current_sr
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for idx in range(num_edges):
                # Try flipping edge idx
                trial_signs = current_signs.copy()
                trial_signs[idx] *= -1
                
                sr, valid = evaluate(trial_signs)
                if valid and sr < best_local_sr - 1e-10:
                    current_signs = trial_signs
                    best_local_sr = sr
                    improved = True
        
        return current_signs, best_local_sr

    # Do local search from best initial solution
    if best_signs is not None:
        max_time = min(deadline - 2, start_time + 40)
        improved_signs, improved_sr = local_search(best_signs, max_time)
        if improved_sr < best_radius:
            best_radius = improved_sr
            best_signs = improved_signs

    # Try more random restarts if time permits
    num_restarts = 0
    while time.time() < deadline - 5:
        signs = rng.choice([-1, 1], size=num_edges)
        sr, valid = evaluate(signs)
        if valid:
            # Local search from this start
            max_time = min(deadline - 3, time.time() + 3)
            improved_signs, improved_sr = local_search(signs, max_time)
            if improved_sr < best_radius:
                best_radius = improved_sr
                best_signs = improved_signs
        num_restarts += 1

    # Final polish: try pair flips
    if best_signs is not None and time.time() < deadline - 2:
        current_signs = best_signs.copy()
        current_sr = best_radius
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(num_edges):
                if time.time() >= deadline - 1:
                    break
                for j in range(i + 1, num_edges):
                    trial_signs = current_signs.copy()
                    trial_signs[i] *= -1
                    trial_signs[j] *= -1
                    sr, valid = evaluate(trial_signs)
                    if valid and sr < current_sr - 1e-10:
                        current_signs = trial_signs
                        current_sr = sr
                        improved = True

        if current_sr < best_radius:
            best_radius = current_sr
            best_signs = current_signs

    # If we didn't find a valid solution, use the baseline
    if best_signs is None:
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)

    return best_signs
