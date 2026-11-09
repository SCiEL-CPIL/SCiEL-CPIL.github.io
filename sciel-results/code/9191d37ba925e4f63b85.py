import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)

    # Identify edge types: top cycle, bottom cycle, matching
    top_cycle = set()
    bottom_cycle = set()
    matching = set()
    for i in range(m):
        a, b = i, (i + 1) % m
        top_cycle.add((a, b))
        top_cycle.add((b, a))
        a, b = m + i, m + (i + 1) % m
        bottom_cycle.add((a, b))
        bottom_cycle.add((b, a))
        a, b = i, i + m
        matching.add((a, b))
        matching.add((b, a))

    def edge_type(i):
        u, v = edges[i]
        if (u, v) in matching:
            return 'match'
        elif u < m:
            return 'top'
        else:
            return 'bottom'

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Find a spanning tree that includes all matching edges.
    # This guarantees the lift is connected: the matching edges (+1)
    # connect the two sides, and the tree edges connect within sides.
    tree_edges = set()
    visited = [False] * n
    # Start with all matching edges (they form a perfect matching, no cycles)
    for i in range(e_count):
        u, v = edges[i]
        if (u, v) in matching:
            tree_edges.add(i)
    # Mark endpoints of matching edges as "connected" via matching
    # Now BFS/DFS to add cycle edges to complete the spanning tree
    # Use union-find to track connectivity
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # Union matching edges
    for i in range(e_count):
        u, v = edges[i]
        if (u, v) in matching:
            union(u, v)

    # Add cycle edges to complete spanning tree
    for i in range(e_count):
        if i in tree_edges:
            continue
        u, v = edges[i]
        if find(u) != find(v):
            tree_edges.add(i)
            union(u, v)

    # Verify we have a spanning tree
    assert len(tree_edges) == n - 1, f"Spanning tree has {len(tree_edges)} edges, expected {n-1}"

    chords = [i for i in range(e_count) if i not in tree_edges]
    num_chords = len(chords)

    best_radius = float('inf')
    best_signs = np.ones(e_count, dtype=int)

    def compute_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    def trace_a2(signs):
        # Tr(A^2) = sum over edges of 2 * (sum of neighbor edge signs)
        # For 3-regular: Tr(A^2) = 2 * sum over all pairs of adjacent edges
        # = 2 * sum_v (sum of edge signs incident to v)^2 - 2 * sum_v (degree_v)
        # Actually Tr(A^2) = sum_{i,j} A_{ij} A_{ji} = sum_{i,j} A_{ij}^2 = 2 * e_count
        # Wait, that's not right. Tr(A^2) = sum_i (A^2)_{ii} = sum_i sum_j A_{ij} A_{ji}
        # For symmetric A: Tr(A^2) = sum_{i,j} A_{ij} A_{ji} = sum_{i,j} A_{ij}^2
        # For signed adjacency with entries in {-1, 0, 1}: A_{ij}^2 = 1 if edge, 0 otherwise
        # So Tr(A^2) = 2 * e_count always. That's not useful.
        #
        # Let me reconsider. Tr(A^2) = sum_{i,j} A_{ij} A_{ji}.
        # For undirected signed graph, A is symmetric, so A_{ji} = A_{ij}.
        # Tr(A^2) = sum_{i,j} A_{ij}^2 = number of edges * 2 = 2*e_count.
        # This is constant! Not useful as a proxy.
        #
        # Better proxy: use the second moment of eigenvalues, or use
        # the fact that for small spectral radius, the graph should be
        # "anti-regular" in some sense.
        #
        # Alternative: use sum of squares of eigenvalues = Tr(A^2) = constant.
        # Use sum of 4th powers = Tr(A^4) as a proxy? Tr(A^4) counts
        # closed walks of length 4. Minimizing Tr(A^4) may help.
        #
        # Tr(A^4) = sum_{i,j,k,l} A_{ij} A_{jk} A_{kl} A_{li}
        # This is expensive to compute directly. Let's use a different approach.
        #
        # Actually, let's just use the actual spectral radius for local search,
        # but limit the number of evaluations. Or use a cheaper proxy.
        #
        # A cheap proxy: for each vertex, compute the product of signs of
        # edges around each triangle. In a prism graph, there are m squares
        # (i, i+1, i+m+1, i+m) and triangles don't exist.
        # The prism has m 4-cycles. The sign product around each 4-cycle
        # determines the eigenvalue contribution.
        #
        # Let's just do local search with actual spectral radius but limit iterations.
        return 0  # placeholder

    # Phase 1: Exhaustive search over chord signs
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break

        signs = np.ones(e_count, dtype=int)
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1

        radius = compute_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Phase 2: Local search polish - try flipping each edge and keep improvements
    # Use a time-limited local search
    current_signs = best_signs.copy()
    current_radius = best_radius
    improved = True
    max_local_iters = 20000

    for iteration in range(max_local_iters):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        for i in range(e_count):
            if time.time() > deadline:
                break
            # Try flipping edge i
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            radius = compute_radius(current_signs)
            if radius < current_radius - 1e-12:
                current_radius = radius
                improved = True
            else:
                current_signs[i] = old_sign

    if current_radius < best_radius:
        best_radius = current_radius
        best_signs = current_signs.copy()

    # Phase 3: Multiple restarts of local search from random perturbations
    for restart in range(20):
        if time.time() > deadline:
            break
        trial_signs = best_signs.copy()
        # Randomly flip a few edges
        num_flips = np.random.default_rng(restart).integers(1, 5)
        flip_indices = np.random.default_rng(restart).choice(e_count, size=num_flips, replace=False)
        for idx in flip_indices:
            trial_signs[idx] = -trial_signs[idx]

        trial_radius = compute_radius(trial_signs)
        if trial_radius < current_radius:
            current_signs = trial_signs.copy()
            current_radius = trial_radius

        # Local search from this starting point
        local_improved = True
        local_iters = 0
        while local_improved and local_iters < 5000:
            if time.time() > deadline:
                break
            local_improved = False
            local_iters += 1
            for i in range(e_count):
                if time.time() > deadline:
                    break
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                radius = compute_radius(current_signs)
                if radius < current_radius - 1e-12:
                    current_radius = radius
                    local_improved = True
                else:
                    current_signs[i] = old_sign

        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    return best_signs
