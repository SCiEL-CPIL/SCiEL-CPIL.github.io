import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}. Maximize n.
    """
    deadline = time.time() + 140  # safety margin before 150s kill

    # Track the best valid graph found (highest n, then highest min degree)
    best_n = 39
    best_min_deg = 0
    best_edges = []

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        required = n - 39

        # Strategy A: Polarity-graph seed + local search
        seed_adj = _polarity_seed(n)
        if seed_adj is not None:
            result = _local_search(seed_adj, n, required, deadline)
            if result is not None:
                n_found, min_deg, edges_found = result
                if n_found > best_n or (n_found == best_n and min_deg > best_min_deg):
                    best_n = n_found
                    best_min_deg = min_deg
                    best_edges = edges_found
                if min_deg >= required:
                    return {'n': n_found, 'edges': edges_found}

        # Strategy B: Greedy from scratch + local search (multiple seeds)
        for seed in range(30):
            if time.time() > deadline:
                break
            adj = _greedy_seed(n, seed)
            result = _local_search(adj, n, required, deadline)
            if result is not None:
                n_found, min_deg, edges_found = result
                if n_found > best_n or (n_found == best_n and min_deg > best_min_deg):
                    best_n = n_found
                    best_min_deg = min_deg
                    best_edges = edges_found
                if min_deg >= required:
                    return {'n': n_found, 'edges': edges_found}
            if time.time() > deadline:
                break

    # Fallback
    return {'n': best_n, 'edges': best_edges}


def _polarity_seed(n):
    """
    Build a polarity graph of the projective plane of order q (q a prime power)
    and trim/pad to n vertices. Polarity graphs are C4-free.
    """
    # Find the largest prime power q such that q^2 + q + 1 <= n
    def is_prime_power(m):
        if m < 2:
            return False
        p = 2
        while p * p <= m:
            if m % p == 0:
                return True
            p += 1
        return True  # m is prime

    best_q = 1
    for q in range(2, 30):
        if not is_prime_power(q):
            continue
        if q * q + q + 1 <= n:
            best_q = q

    if best_q < 2:
        return None

    q = best_q
    N = q * q + q + 1

    # Build projective plane points and lines
    points = []
    for a in range(q + 1):
        for b in range(q):
            points.append((a, b))
    points.append((0, 1))

    # Build lines: for each slope a, lines y = a*x + b, plus vertical line x=0
    # Standard construction:
    lines = []
    for a in range(q + 1):
        for b in range(q):
            line = []
            for x in range(q):
                if a == q:
                    y = x + b
                else:
                    y = (a * x + b) % q
                line.append((a, y))
            lines.append(line)
    # Vertical line x=0: points (0, b) for b in 0..q-1, plus (0,1)
    lines.append([(0, b) for b in range(q)] + [(0, 1)])

    # Map points to indices
    point_index = {p: i for i, p in enumerate(points)}

    # Polarity: point p is incident to line L_p
    # Define a polarity: (a, b) -> line with equation... 
    # Standard polarity: (a, b) is incident to line a*x - b = 0 (or similar)
    # Let's use: point (a, b) is incident to the line that contains it with "dual" property
    # Simpler: use the standard polarity of PG(2,q)
    # Point (a,b) is incident to line: for (a,b), the line is the set of points (x,y) with a*x - b*y = 0 (if q is odd)
    # This is getting complex. Let me use a simpler known C4-free construction.

    # Alternative: use the incidence graph approach or just return a random C4-free graph
    # Let's use a simpler approach: greedy construction with a good heuristic

    # Actually, let me just return None and rely on greedy + local search
    return None


def _greedy_seed(n, seed):
    """
    Greedily build a C4-free graph by adding edges in random order,
    checking C4 constraint via vectorized method.
    """
    random.seed(seed)
    adj = np.zeros((n, n), dtype=bool)
    degrees = np.zeros(n, dtype=int)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)

    for u, v in all_edges:
        if not adj[u, v] and not _has_c4(adj, u, v):
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1

    return adj


def _has_c4(adj, u, v):
    """
    Vectorized check: does adding edge (u,v) create a C4?
    A C4 exists iff there is x != u,v with adj[u,x] and adj[x,y] and adj[y,v] for some y != u,v,x.
    Equivalent: (N(u) \\ {v}) has a vertex x such that N(x) \\ {u,v} intersects N(v).
    Vectorized: for all x in N(u)\\{v}, check if adj[x, N(v)\\{u,v}] has any True.
    """
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]

    # x candidates: neighbors of u, excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    x_candidates = np.where(mask_u)[0]

    if len(x_candidates) == 0:
        return False

    # For each x in x_candidates, check if N(x) intersects N(v) \\ {u, v}
    # adj[x_candidates, :] gives the adjacency rows for all x_candidates
    # We need: for some x, adj[x, y] for some y in N(v) \\ {u, v}
    
    # Create a mask for y in N(v) \\ {u, v}
    mask_v = adj[v].copy()
    mask_v[u] = False
    mask_v[v] = False

    # For each x in x_candidates, check if adj[x, mask_v] has any True
    # adj[x_candidates, :] is a (len(x_candidates), n) boolean array
    # We want: any row has a True in the columns where mask_v is True
    sub = adj[x_candidates, :]  # (len(x_candidates), n)
    # For each row, check if there's a True in the mask_v columns
    result = sub[:, mask_v]  # (len(x_candidates), count of mask_v True)
    return bool(np.any(result))


def _local_search(adj, n, required, deadline):
    """
    Local search to maximize minimum degree while maintaining C4-freeness.
    Returns (n, min_degree, edges) or None if no improvement.
    """
    adj = adj.copy()
    degrees = adj.sum(axis=1).astype(int)
    best_min_deg = int(degrees.min())
    best_adj = adj.copy()

    # Repeatedly try to add an edge that increases the minimum degree
    for iteration in range(20000):
        if time.time() > deadline:
            break

        min_deg = int(degrees.min())
        if min_deg >= required:
            break

        # Find vertices with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        random.shuffle(min_vertices.tolist())

        improved = False
        for u in min_vertices:
            if time.time() > deadline:
                break

            # Try adding edges from u to non-neighbors, sorted by target degree
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            if len(non_neighbors) == 0:
                continue

            # Sort by degree of target (ascending)
            order = non_neighbors[np.argsort(degrees[non_neighbors])]

            for v in order:
                if time.time() > deadline:
                    break
                if not _has_c4(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
                    break

            if improved:
                break

        if not improved:
            # Try edge swaps: remove edge (a,b) where both a,b have high degree,
            # add edge (u,c) where u has low degree
            # This is complex; just try random edge additions
            # Try adding any edge that doesn't create C4 and increases some degree
            all_missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            if len(all_missing) == 0:
                break
            random.shuffle(all_missing)
            found = False
            for u, v in all_missing[:50]:
                if time.time() > deadline:
                    break
                if not _has_c4(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
                    found = True
                    break
            if not found:
                break

        # Update best
        cur_min = int(degrees.min())
        if cur_min > best_min_deg:
            best_min_deg = cur_min
            best_adj = adj.copy()

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])

    return (n, best_min_deg, edges)

# EVOLVE-BLOCK-END
