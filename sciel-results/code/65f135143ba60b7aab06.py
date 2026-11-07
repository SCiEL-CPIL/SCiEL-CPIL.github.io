"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    total_nodes = 2 * n_vertices

    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)

    # Precompute which edges are cycle edges vs matching edges
    # Cycle edges: (i, i+1 mod m) for top, (m+i, m+i+1 mod m) for bottom
    # Matching edges: (i, i+m)
    edge_type = np.zeros(n_edges, dtype=np.int8)  # 0=matching, 1=top cycle, 2=bottom cycle
    for idx in range(n_edges):
        u = edge_u[idx]
        v = edge_v[idx]
        if abs(u - v) == m:
            edge_type[idx] = 0
        elif u < m and v < m:
            edge_type[idx] = 1
        elif u >= m and v >= m:
            edge_type[idx] = 2

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

    def spectral_radius_exact(signs):
        """Compute exact spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def spectral_radius_power(signs, n_iter=30):
        """Fast power iteration estimate of spectral radius."""
        A = build_signed_adj(signs)
        x = np.ones(n_vertices, dtype=np.float64)
        x /= np.linalg.norm(x)
        for _ in range(n_iter):
            x = A @ x
            norm = np.linalg.norm(x)
            if norm < 1e-15:
                return 0.0
            x /= norm
        return float(np.abs(x @ A @ x))

    def is_lift_connected(signs):
        """Fast connectivity check for the 2-lift using BFS."""
        # Build adjacency for the lift graph
        # Each base vertex v has neighbors in lift: for each edge (v,w) with sign s,
        # (v,0) connects to (w,0) if s=+1 or (w,1) if s=-1
        # (v,1) connects to (w,1) if s=+1 or (w,0) if s=-1
        # Precompute base adjacency
        base_adj = [[] for _ in range(n_vertices)]
        for idx in range(n_edges):
            u = edge_u[idx]
            v = edge_v[idx]
            s = int(signs[idx])
            base_adj[u].append((v, s))
            base_adj[v].append((u, s))

        visited = np.zeros(total_nodes, dtype=np.bool_)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            v = node // 2
            side = node % 2
            for (w, s) in base_adj[v]:
                if s == 1:
                    neighbor = w * 2 + side
                else:
                    neighbor = w * 2 + (1 - side)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)

        return count == total_nodes

    def evaluate(signs, use_power=True):
        """Check validity and compute radius. Returns (valid, radius)."""
        if not is_lift_connected(signs):
            return False, float('inf')
        if use_power:
            radius = spectral_radius_power(signs)
        else:
            radius = spectral_radius_exact(signs)
        return True, radius

    # Generate structured initial patterns based on cycle frustration
    def generate_structured_patterns():
        """Generate sign patterns that frustrate cycles for low spectral radius."""
        patterns = []
        # Find indices of edges by type
        top_cycle_edges = [i for i in range(n_edges) if edge_type[i] == 1]
        bottom_cycle_edges = [i for i in range(n_edges) if edge_type[i] == 2]
        matching_edges = [i for i in range(n_edges) if edge_type[i] == 0]

        # Pattern 1: All +1 (trivial, may not be connected)
        p = np.ones(n_edges, dtype=np.float64)
        patterns.append(p)

        # Pattern 2: Alternating on top cycle, all +1 elsewhere
        for phase in range(2):
            p = np.ones(n_edges, dtype=np.float64)
            for k, idx in enumerate(top_cycle_edges):
                if (k + phase) % 2 == 1:
                    p[idx] = -1
            patterns.append(p)

        # Pattern 3: Alternating on bottom cycle
        for phase in range(2):
            p = np.ones(n_edges, dtype=np.float64)
            for k, idx in enumerate(bottom_cycle_edges):
                if (k + phase) % 2 == 1:
                    p[idx] = -1
            patterns.append(p)

        # Pattern 4: Alternating on both cycles
        for phase in range(4):
            p = np.ones(n_edges, dtype=np.float64)
            for k, idx in enumerate(top_cycle_edges):
                if (k + phase) % 2 == 1:
                    p[idx] = -1
            for k, idx in enumerate(bottom_cycle_edges):
                if (k + phase // 2) % 2 == 1:
                    p[idx] = -1
            patterns.append(p)

        # Pattern 5: Frustrate matching edges
        for phase in range(2):
            p = np.ones(n_edges, dtype=np.float64)
            for k, idx in enumerate(matching_edges):
                if (k + phase) % 2 == 1:
                    p[idx] = -1
            patterns.append(p)

        # Pattern 6: Mixed frustration
        for phase in range(4):
            p = np.ones(n_edges, dtype=np.float64)
            for k, idx in enumerate(top_cycle_edges):
                if (k + phase) % 2 == 1:
                    p[idx] = -1
            for k, idx in enumerate(matching_edges):
                if (k + phase // 2) % 2 == 1:
                    p[idx] = -1
            patterns.append(p)

        return patterns

    # Initial structured search
    rng = np.random.default_rng(42)

    best_signs = None
    best_radius = float('inf')
    best_valid = False

    # Phase 1: Try structured patterns
    for pattern in generate_structured_patterns():
        if time.time() - start_time > DEADLINE:
            break
        valid, radius = evaluate(pattern, use_power=True)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = pattern.copy()
            best_valid = True

    # Phase 2: Broad random search with power iteration screening
    n_random = 300
    for i in range(n_random):
        if time.time() - start_time > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
        valid, radius = evaluate(signs, use_power=True)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            best_valid = True

    # Phase 3: Local search from best found (1-flip)
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for flip_iter in range(300):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs, use_power=True)

                if valid and new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Phase 4: 2-flip local search from best
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for flip_iter in range(100):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() - start_time > DEADLINE:
                        break
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    valid, new_radius = evaluate(signs, use_power=True)
                    if valid and new_radius < best_radius - 1e-10:
                        best_radius = new_radius
                        best_signs = signs.copy()
                        best_valid = True
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

    # Phase 5: Try more random starts and local search
    for init_idx in range(50):
        if time.time() - start_time > DEADLINE:
            break

        signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
        valid, radius = evaluate(signs, use_power=True)
        if valid:
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                best_valid = True

        # Quick local search from this start
        for flip_iter in range(80):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                valid, new_radius = evaluate(signs, use_power=True)

                if valid and new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Phase 6: Final polish with exact eigenvalues
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        best_radius_exact = spectral_radius_exact(signs)
        best_radius = best_radius_exact

        for polish_iter in range(50):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break

                signs[i] = -signs[i]
                if not is_lift_connected(signs):
                    signs[i] = -signs[i]
                    continue
                new_radius_exact = spectral_radius_exact(signs)
                if new_radius_exact < best_radius - 1e-12:
                    best_radius = new_radius_exact
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]

            if not improved:
                break

    # Return as int array
    if best_signs is None:
        return np.ones(n_edges, dtype=int)

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
