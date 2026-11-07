"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 80  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)

    # Identify edge types: cycle1 (0..m-1), cycle2 (m..2m-1), matching (2m..3m-1)
    # Based on the problem statement: two m-cycles on 0..m-1 and m..2m-1, joined by matching (i, i+m)

    def build_signed_adj(signs):
        """Build the 2m x 2m signed adjacency matrix."""
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx in range(n_edges):
            s = signs[idx]
            u = edge_u[idx]
            v = edge_v[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def is_lift_connected_fast(signs):
        """Fast connectivity check for the 2-lift using BFS."""
        n = n_vertices
        adj = [[] for _ in range(2 * n)]
        for idx in range(n_edges):
            s = signs[idx]
            u = edge_u[idx]
            v = edge_v[idx]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)

        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)

        return count == 2 * n

    def evaluate(signs):
        """Check validity and compute radius. Returns (valid, radius)."""
        if not is_lift_connected_fast(signs):
            return False, float('inf')
        radius = spectral_radius(signs)
        return True, radius

    rng = np.random.default_rng(42)

    best_signs = None
    best_radius = float('inf')
    best_valid = False

    def try_signs(signs):
        """Try a sign assignment and update best if valid and better."""
        nonlocal best_signs, best_radius, best_valid
        valid, radius = evaluate(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            best_valid = True
            return True
        return False

    # Key insight: For the 2-lift to be connected, the holonomy (product of signs around any cycle) 
    # must be -1 for at least one fundamental cycle. The prism graph has fundamental cycles:
    # - Cycle 1 (bottom): product of signs on edges 0..m-1
    # - Cycle 2 (top): product of signs on edges m..2m-1
    # - Matching + cycle segments: various combinations
    
    # To minimize spectral radius, we want the signed adjacency matrix to have small eigenvalues.
    # A good strategy: make each cycle have an odd number of -1 signs (product = -1),
    # which "frustrates" the cycle and reduces the spectral radius.

    def generate_frustrated_patterns():
        """Generate patterns where each cycle has odd number of -1 signs."""
        patterns = []
        
        # Strategy: Place -1 signs to minimize spectral radius
        # For a cycle of length m, the optimal placement depends on m
        
        # Pattern A: One -1 per cycle, placed to minimize max eigenvalue
        # For cycle 1: place -1 at position p1
        # For cycle 2: place -1 at position p2
        # Matching: all +1 or some -1
        
        for p1 in range(m):
            for p2 in range(m):
                signs = np.ones(n_edges, dtype=np.float64)
                signs[p1] = -1  # One -1 in cycle 1
                signs[m + p2] = -1  # One -1 in cycle 2
                patterns.append(signs.copy())
        
        # Pattern B: Three -1 per cycle (for odd m, this gives product = -1)
        if m % 2 == 1:
            for p1 in range(m):
                for p2 in range(m):
                    signs = np.ones(n_edges, dtype=np.float64)
                    # Place three -1s in cycle 1, spread out
                    signs[p1] = -1
                    signs[(p1 + m // 3) % m] = -1
                    signs[(p1 + 2 * m // 3) % m] = -1
                    signs[m + p2] = -1
                    signs[m + (p2 + m // 3) % m] = -1
                    signs[m + (p2 + 2 * m // 3) % m] = -1
                    patterns.append(signs.copy())
        
        # Pattern C: Matching also frustrated
        for p1 in range(m):
            for p2 in range(m):
                for pm in range(m):
                    signs = np.ones(n_edges, dtype=np.float64)
                    signs[p1] = -1
                    signs[m + p2] = -1
                    signs[2 * m + pm] = -1
                    patterns.append(signs.copy())
        
        # Pattern D: Random frustrated configurations
        for _ in range(50):
            signs = np.ones(n_edges, dtype=np.float64)
            # Ensure odd number of -1 in cycle 1
            n_neg1 = rng.choice([1, 3, 5]) if m % 2 == 1 else rng.choice([1, 3])
            n_neg1 = min(n_neg1, m)
            if n_neg1 % 2 == 0:
                n_neg1 += 1
            n_neg1 = min(n_neg1, m)
            neg_positions1 = rng.choice(m, size=n_neg1, replace=False)
            for p in neg_positions1:
                signs[p] = -1
            
            # Ensure odd number of -1 in cycle 2
            n_neg2 = rng.choice([1, 3, 5]) if m % 2 == 1 else rng.choice([1, 3])
            n_neg2 = min(n_neg2, m)
            if n_neg2 % 2 == 0:
                n_neg2 += 1
            n_neg2 = min(n_neg2, m)
            neg_positions2 = rng.choice(m, size=n_neg2, replace=False)
            for p in neg_positions2:
                signs[m + p] = -1
            
            # Matching: random
            for i in range(m):
                if rng.random() < 0.3:
                    signs[2 * m + i] = -1
            
            patterns.append(signs.copy())
        
        return patterns

    # Try frustrated patterns first
    for signs in generate_frustrated_patterns():
        if time.time() - start_time > DEADLINE:
            break
        try_signs(signs)

    # Also try all +1 as baseline
    try_signs(np.ones(n_edges, dtype=np.float64))

    # Local search: single flip optimization
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        
        # Multiple passes of local search
        for flip_iter in range(50):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            # Try flipping each edge
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs)

                if valid and new_radius < best_radius - 1e-12:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # If no valid lift, try random
    if not best_valid:
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
            try_signs(signs)

    # Final polish with tighter tolerance
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs)
                if valid and new_radius < best_radius - 1e-14:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]
            if not improved:
                break

    if best_signs is None:
        return np.ones(n_edges, dtype=int)

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
