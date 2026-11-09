"""Voronoi-based topological heuristic combined with reference-tour embedding:
identify spatially isolated regions via Voronoi cell adjacency, assign high
attractiveness to edges that bridge disjoint regions, and strongly promote
edges from a high-quality reference tour found via multi-start NN + 2-opt."""

import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import linear_sum_assignment

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

    # Recover 2D coordinates from distance matrix via classical MDS
    try:
        D2 = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1][:2]
        coords = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))
        
        # Compute Voronoi diagram
        vor = Voronoi(coords)
        
        # Build adjacency from Voronoi ridge points
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
        voronoi_heur = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if region_id[i] != region_id[j]:
                    voronoi_heur[i, j] = 1000.0 / max(D[i, j], 1e-9)
                else:
                    voronoi_heur[i, j] = 1.0 / max(D[i, j], 1e-9)
        
    except Exception:
        # Fallback: simple inverse distance
        voronoi_heur = 1.0 / np.maximum(D, 1e-9)
        voronoi_heur[np.diag_indices(n)] = 0.0
        region_id = np.zeros(n, dtype=int)

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
    num_starts = 300
    for _ in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        s = int(rng.integers(n))
        refine(nearest_neighbor(s))

    # Random perturbation + local search on best tour
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        t = list(best_tour)
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

    # Combine Voronoi heuristic with reference tour embedding + Hungarian reduced-cost landscape
    diag = np.eye(n, dtype=bool)
    base = voronoi_heur.copy()
    
    # Hungarian reduced-cost landscape: compute reduced costs from optimal assignment
    try:
        # For TSP, we want a 1-regular directed graph. Use Hungarian on a modified cost
        # matrix to get a permutation, then compute reduced costs as a smooth signal.
        # Use distance matrix directly (symmetric), solve assignment problem.
        row_ind, col_ind = linear_sum_assignment(D)
        # Build the assignment tour (may have subtours, but gives us a reference)
        assign_tour = np.zeros(n, dtype=int)
        for r, c in zip(row_ind, col_ind):
            assign_tour[r] = c
        # Compute reduced costs: D[i,j] - u[i] - v[j] where u,v are dual variables
        # Approximate duals from the assignment: u[i] = D[i, assign_tour[i]] - v[assign_tour[i]]
        # Simpler: use the assignment cost as a baseline and measure regret
        assign_costs = D[row_ind, col_ind]
        mean_assign_cost = np.mean(assign_costs)
        
        # Reduced cost landscape: lower regret = more attractive
        # regret[i,j] = D[i,j] - (D[i, assign_tour[i]] + D[assign_tour[j], j]) / 2
        # This measures how much worse edge (i,j) is compared to assignment edges
        row_costs = D[np.arange(n), assign_tour]  # D[i, assign_tour[i]]
        col_costs = D[assign_tour, np.arange(n)]  # D[assign_tour[j], j]
        
        # Smooth reduced cost: use soft version
        regret = D - (row_costs[:, None] + col_costs[None, :]) / 2.0
        # Larger regret = worse edge, so attractiveness = exp(-regret / scale)
        scale = max(np.std(regret), 1e-9)
        hungarian_heur = np.exp(-np.maximum(regret, 0) / scale)
        
        # Normalize to be comparable
        hmax = np.max(hungarian_heur[~diag])
        if hmax > 1e-12:
            hungarian_heur = hungarian_heur / hmax
        
        # Blend: Hungarian provides global gradient, Voronoi provides local structure
        base = 0.6 * base + 0.4 * (100.0 * hungarian_heur)
        
    except Exception:
        pass
    
    # Strongly promote reference tour edges
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base[i, j] = max(base[i, j], 1.0e5)

    base[diag] = 1e-9
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)

    return base
