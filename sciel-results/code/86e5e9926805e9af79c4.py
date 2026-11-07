import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    n = graph['n']
    m = n // 2
    edges = graph['edges']

    # Classify edges: top cycle, bottom cycle, matching
    top_cycle = []   # edges among 0..m-1
    bottom_cycle = []  # edges among m..2m-1
    matching = []    # edges (i, i+m)
    edge_index = {}  # (u,v) or (v,u) -> index in edges list

    for idx, (u, v) in enumerate(edges):
        key = (u, v) if u < v else (v, u)
        edge_index[key] = idx
        if u < m and v < m:
            top_cycle.append((idx, u, v))
        elif u >= m and v >= m:
            bottom_cycle.append((idx, u, v))
        else:
            matching.append((idx, u, v))

    # Build cycle adjacency: for each cycle, map vertex -> list of (neighbor, edge_idx)
    top_adj = {i: [] for i in range(m)}
    for idx, u, v in top_cycle:
        top_adj[u].append((v, idx))
        top_adj[v].append((u, idx))

    bottom_adj = {i: [] for i in range(m)}
    for idx, u, v in bottom_cycle:
        iu = u - m
        iv = v - m
        bottom_adj[iu].append((iv, idx))
        bottom_adj[iv].append((iu, idx))

    # Matching: for each i in 0..m-1, find edge (i, i+m)
    match_edge_idx = {}
    for idx, u, v in matching:
        if u < m:
            match_edge_idx[u] = idx
        else:
            match_edge_idx[v - m] = idx

    # Signs array
    signs = np.zeros(len(edges), dtype=np.int8)

    # Precompute: for each vertex in each cycle, the two neighbor vertices and their edge indices
    top_v = []
    for i in range(m):
        nbrs = top_adj[i]
        top_v.append((nbrs[0][0], nbrs[0][1], nbrs[1][0], nbrs[1][1]))

    bottom_v = []
    for i in range(m):
        nbrs = bottom_adj[i]
        bottom_v.append((nbrs[0][0], nbrs[0][1], nbrs[1][0], nbrs[1][1]))

    # Enumerate all 2^(2m) sign assignments for cycle edges
    # top signs: 2^m, bottom signs: 2^m
    total_top = 1 << m
    total_bottom = 1 << m

    best_radius = float('inf')
    best_top = 0
    best_bottom = 0

    # For each top pattern, precompute the top contribution row-wise
    # A_eff[i] = c_top_row[i] + c_bottom_row[i]
    # c_top_row[i] has nonzeros at top_v[i][0] and top_v[i][2] with signs s_top[edge]
    # c_bottom_row[i] has nonzeros at bottom_v[i][0] and bottom_v[i][2] with signs s_bottom[edge]

    # Spectral radius of symmetric matrix via eigvalsh
    # We build the m x m matrix A_eff and compute max |eigenvalue|

    def compute_radius(top_bits, bottom_bits):
        # Build A_eff
        A = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            # top neighbors
            n1, e1, n2, e2 = top_v[i]
            s1 = 1.0 if (top_bits >> e1) & 1 else -1.0
            s2 = 1.0 if (top_bits >> e2) & 1 else -1.0
            A[i, n1] += s1
            A[i, n2] += s2
            # bottom neighbors
            n1b, e1b, n2b, e2b = bottom_v[i]
            s1b = 1.0 if (bottom_bits >> e1b) & 1 else -1.0
            s2b = 1.0 if (bottom_bits >> e2b) & 1 else -1.0
            A[i, n1b] += s1b
            A[i, n2b] += s2b
        # Symmetrize (should already be symmetric but ensure)
        A = (A + A.T) * 0.5
        # Spectral radius
        try:
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
        except np.linalg.LinAlgError:
            return float('inf')
        return radius

    # Enumerate
    for top_bits in range(total_top):
        if time.time() > DEADLINE:
            break
        # Compute top product
        top_prod = 1
        for i in range(m):
            _, e1, _, e2 = top_v[i]
            # Each edge counted twice in the cycle, but product over vertices
            # Actually, product of signs around the cycle:
            # The product of all signs on edges in the top cycle
            # Each edge appears in exactly 2 vertices' adjacency, so product over vertices of (product of edge signs at vertex) = product over edges of (sign^2) = 1
            # That's not what we want. We want the product of signs along the cycle edges.
            pass

        # Product of signs around top cycle = product over all top cycle edges of sign
        # Each top cycle edge has a sign; the cycle product is the product of all m top edge signs
        # But we need to be careful: the "cycle product" in the lift context
        # The lift is connected iff the holonomy around the top cycle != holonomy around bottom cycle
        # Holonomy = product of signs along the cycle

        # For a cycle with edges e_0, e_1, ..., e_{m-1} in order, the holonomy is s_{e_0} * s_{e_1} * ... * s_{e_{m-1}}
        # We need to compute this product efficiently

        # Precompute: for each top_bits, the product of all top edge signs
        # top_prod = product over all top cycle edges of sign
        # Since each edge is in exactly one position, and there are m edges:
        top_prod = 1
        for idx, u, v in top_cycle:
            if (top_bits >> idx) & 1:
                top_prod *= 1
            else:
                top_prod *= -1

        for bottom_bits in range(total_bottom):
            if time.time() > DEADLINE:
                break

            # Bottom product
            bottom_prod = 1
            for idx, u, v in bottom_cycle:
                if (bottom_bits >> idx) & 1:
                    bottom_prod *= 1
                else:
                    bottom_prod *= -1

            # Connectivity: top_prod != bottom_prod
            if top_prod == bottom_prod:
                continue

            radius = compute_radius(top_bits, bottom_bits)
            if radius < best_radius:
                best_radius = radius
                best_top = top_bits
                best_bottom = bottom_bits

    # Now construct the final signs
    # Set cycle edge signs
    for idx, u, v in top_cycle:
        signs[idx] = 1 if (best_top >> idx) & 1 else -1

    for idx, u, v in bottom_cycle:
        signs[idx] = 1 if (best_bottom >> idx) & 1 else -1

    # Gauge-fix: set matching signs to +1
    # But wait: the gauge transformation changes the effective matrix.
    # Let me reconsider.

    # Actually, the spectral radius I computed is for the matrix A_eff = C_top + C_bottom
    # where C_top and C_bottom are the signed cycle adjacency matrices.
    # The actual 2m x 2m signed adjacency matrix is:
    # A = [C_top, M; M^T, C_bottom]
    # where M is the signed matching.
    #
    # By gauge transformation: flip vertex (i, 1) by g_i for i in 0..m-1
    # This changes:
    # - Matching edge (i, i+m): sign becomes s_match * g_i * 1 (flipping one endpoint)
    #   Actually, flipping vertex (i,1) by g_i: the edge (i, i+m) connects (i,0) and (i+m,1)
    #   Wait, I need to be more careful with the lift construction.

    # Let me reconsider the lift construction.
    # Base graph vertices: 0..2m-1
    # Lift vertices: (v, 0) and (v, 1) for each base vertex v
    # Edge (u, v) with sign s:
    #   s = +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   s = -1: (u,0)-(v,1) and (u,1)-(v,0)

    # The signed adjacency matrix of the base graph A_s satisfies:
    # A_s^2 has diagonal entries = degree = 3
    # The lift adjacency matrix B (size 2n x 2n) satisfies B^2 = A_s^2 (in some sense)
    # Actually, the eigenvalues of the lift are the eigenvalues of A_s, each with multiplicity 2? No.

    # The correct relationship: the lift is connected and 3-regular iff the sign assignment
    # makes the associated 2-cover connected. The eigenvalues of the lift are the same as
    # the eigenvalues of the signed adjacency matrix A_s.

    # So I just need to minimize the spectral radius of A_s (the 2m x 2m signed adjacency matrix).

    # Let me rebuild: construct A_s directly and compute its spectral radius.

    # Actually, let me just construct the full 2m x 2m signed adjacency matrix and compute
    # its spectral radius directly. This is simpler and avoids gauge complications.

    def compute_full_radius(signs_arr):
        A = np.zeros((2 * m, 2 * m), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = float(signs_arr[idx])
            A[u, v] += s
            A[v, u] += s
        A = (A + A.T) * 0.5
        try:
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
        except np.linalg.LinAlgError:
            return float('inf')
        return radius

    # Check if lift is connected
    def is_connected(signs_arr):
        # Build the lift graph and check connectivity
        # Lift has 2 * 2m = 4m vertices
        # Vertex (v, side) for v in 0..2m-1, side in {0, 1}
        # Edge (u, v) with sign s:
        #   s = +1: (u,0)-(v,0), (u,1)-(v,1)
        #   s = -1: (u,0)-(v,1), (u,1)-(v,0)

        # Use BFS/DFS
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))

        # Build adjacency for the lift
        # For each base edge (u, v) with sign s:
        #   if s = +1: (u,0)~(v,0), (v,0)~(u,0), (u,1)~(v,1), (v,1)~(u,1)
        #   if s = -1: (u,0)~(v,1), (v,1)~(u,0), (u,1)~(v,0), (v,0)~(u,1)

        while stack:
            v, side = stack.pop()
            for (u, w) in [(a, b) for (a, b) in edges if v == a or v == b]:
                # This is inefficient; let me precompute adjacency
                pass

        # More efficient: precompute lift adjacency
        lift_adj = {}
        for idx, (u, v) in enumerate(edges):
            s = signs_arr[idx]
            if s == 1:
                lift_adj.setdefault((u, 0), []).append((v, 0))
                lift_adj.setdefault((v, 0), []).append((u, 0))
                lift_adj.setdefault((u, 1), []).append((v, 1))
                lift_adj.setdefault((v, 1), []).append((u, 1))
            else:
                lift_adj.setdefault((u, 0), []).append((v, 1))
                lift_adj.setdefault((v, 1), []).append((u, 0))
                lift_adj.setdefault((u, 1), []).append((v, 0))
                lift_adj.setdefault((v, 0), []).append((u, 1))

        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))
        while stack:
            v, side = stack.pop()
            for nb in lift_adj.get((v, side), []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        return len(visited) == 4 * m

    # Redo the enumeration with the full matrix
    best_radius = float('inf')
    best_top = 0
    best_bottom = 0

    # Precompute top cycle edge indices and bottom cycle edge indices
    top_edge_idxs = [idx for idx, u, v in top_cycle]
    bottom_edge_idxs = [idx for idx, u, v in bottom_cycle]

    for top_bits in range(total_top):
        if time.time() > DEADLINE:
            break
        # Set top signs
        for idx in top_edge_idxs:
            signs[idx] = 1 if (top_bits >> idx) & 1 else -1

        # Top product
        top_prod = 1
        for idx in top_edge_idxs:
            if signs[idx] == -1:
                top_prod *= -1

        for bottom_bits in range(total_bottom):
            if time.time() > DEADLINE:
                break

            # Set bottom signs
            for idx in bottom_edge_idxs:
                signs[idx] = 1 if (bottom_bits >> idx) & 1 else -1

            # Bottom product
            bottom_prod = 1
            for idx in bottom_edge_idxs:
                if signs[idx] == -1:
                    bottom_prod *= -1

            if top_prod == bottom_prod:
                continue

            # Set matching signs to +1
            for i in range(m):
                signs[match_edge_idx[i]] = 1

            # Check connectivity
            if not is_connected(signs):
                continue

            # Compute spectral radius
            radius = compute_full_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_top = top_bits
                best_bottom = bottom_bits

    # Apply best signs
    for idx in top_edge_idxs:
        signs[idx] = 1 if (best_top >> idx) & 1 else -1
    for idx in bottom_edge_idxs:
        signs[idx] = 1 if (best_bottom >> idx) & 1 else -1
    for i in range(m):
        signs[match_edge_idx[i]] = 1

    # Final verification
    if not is_connected(signs):
        # Fallback: try a few random assignments
        rng = np.random.default_rng(3)
        for _ in range(1000):
            signs[:] = rng.choice([-1, 1], size=len(edges))
            if is_connected(signs):
                break

    return signs
