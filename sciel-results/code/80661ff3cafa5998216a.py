"""Voronoi-based topological heuristic: identify spatially isolated regions via
Voronoi cell adjacency, then assign high attractiveness to edges that bridge
disjoint regions and low attractiveness to intra-region edges. Combine with
reference-tour embedding for cross-region edges."""

import numpy as np
import time
from scipy.spatial import Voronoi

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=300):
        """Iterative 2-opt with early exit on time."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                if i == 0:
                    j_range = range(2, n - 1)
                else:
                    j_range = range(i + 2, n)
                a = t[i]
                b = t[i + 1]
                for j in j_range:
                    c = t[j]
                    d = t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        a = t[i]
                        b = t[i + 1]
                        improved = True
        return t

    def or_opt(t, max_passes=30):
        """Relocate single node to best position."""
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for k in range(n):
                node = t[k]
                prev = t[(k - 1) % n]
                nxt = t[(k + 1) % n]
                cost_remove = D[prev, node] + D[node, nxt] - D[prev, nxt]
                best_delta = cost_remove
                best_pos = -1
                for pos in range(n - 1):
                    if pos == k or pos == (k - 1) % n:
                        continue
                    a = t[pos]
                    b = t[(pos + 1) % n]
                    if a == prev or b == prev or a == nxt or b == nxt:
                        continue
                    cost_insert = D[a, node] + D[node, b] - D[a, b]
                    delta = cost_remove + cost_insert
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_pos = pos
                if best_pos >= 0:
                    t.pop(k)
                    t.insert(best_pos + 1, node)
                    improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    # Extract points from distance matrix (reconstruct from D)
    # For uniform random points in unit square, we can estimate coordinates
    # by solving for positions that produce the given distance matrix
    # Simple approach: use PCA-like reconstruction or assume we can recover points
    # Actually, for this task, the distance matrix IS the input, so we work with it directly
    
    # Compute Voronoi diagram to identify spatial structure
    # We need to recover point coordinates from the distance matrix
    # Use MDS (Multidimensional Scaling) to get 2D coordinates
    try:
        from scipy.spatial.distance import squareform
        # Convert distance matrix to dissimilarity
        # Use classical MDS
        D2 = D ** 2
        # Centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 components
        idx = np.argsort(eigenvalues)[::-1][:2]
        coords = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx])
        
        # Compute Voronoi diagram
        vor = Voronoi(coords)
        
        # Build adjacency: two points are adjacent if their Voronoi cells share an edge
        # vor.ridge_points[i] gives the two point indices whose cells share ridge i
        adjacent = np.zeros((n, n), dtype=bool)
        for i in range(len(vor.ridge_points)):
            p1, p2 = vor.ridge_points[i]
            if p1 != -1 and p2 != -1:
                adjacent[p1, p2] = True
                adjacent[p2, p1] = True
        
        # Find connected components (regions)
        visited = np.zeros(n, dtype=bool)
        regions = []
        for i in range(n):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                region = []
                while stack:
                    v = stack.pop()
                    region.append(v)
                    for u in range(n):
                        if adjacent[v, u] and not visited[u]:
                            visited[u] = True
                            stack.append(u)
                regions.append(region)
        
        # Assign region ID to each point
        region_id = np.zeros(n, dtype=int)
        for r, region in enumerate(regions):
            for node in region:
                region_id[node] = r
        
        # Build Voronoi-based heuristic
        # High attractiveness for edges that bridge different regions
        # Low attractiveness for intra-region edges (to prevent local loops)
        voronoi_heur = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if region_id[i] != region_id[j]:
                    # Cross-region edge: high attractiveness
                    # Scale by inverse distance (shorter bridges are better)
                    voronoi_heur[i, j] = 1000.0 / max(D[i, j], 1e-9)
                else:
                    # Intra-region edge: low attractiveness
                    voronoi_heur[i, j] = 1.0 / max(D[i, j], 1e-9)
        
    except Exception:
        # Fallback: use simple inverse distance
        voronoi_heur = 1.0 / np.maximum(D, 1e-9)
        voronoi_heur[np.diag_indices(n)] = 0.0

    # Multi-start construction + local search for reference tour
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)

    def refine(tour):
        nonlocal best_tour, best_len
        t = two_opt(tour)
        t = or_opt(t)
        t = two_opt(t)
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
        return t

    # Start from node 0
    refine(nearest_neighbor(0))

    # Randomized starts
    num_starts = 200
    for _ in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    for _ in range(60):
        if time.time() > deadline - 5.0:
            break
        t = list(best_tour)
        # Random 2-opt segment reversal
        for _ in range(2):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i > j:
                i, j = j, i
            if j - i >= 1:
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        refine(t)

    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0))

    # Combine Voronoi heuristic with reference tour embedding
    # Reference tour edges that cross regions get boosted
    diag = np.eye(n, dtype=bool)
    base = voronoi_heur.copy()
    
    # Boost reference tour edges that are cross-region
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        if region_id[i] != region_id[j]:
            base[i, j] = max(base[i, j], 5000.0)
        else:
            # Intra-region reference edges: moderate boost
            base[i, j] = max(base[i, j], 100.0)

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
