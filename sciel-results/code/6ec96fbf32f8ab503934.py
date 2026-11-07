"""Heuristic matrix for ACO-TSP: Rank + MST structural prior + strong local-search tour prior
   + geometric (Delaunay/Hull/KNN) boosts with a light crossing penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 330s wall-clock kill; keep 30s safety margin

def _remaining():
    return _DEADLINE - time.time()

def _mds_coordinates(D):
    """Recover 2D coordinates from a Euclidean distance matrix using Classical MDS."""
    n = D.shape[0]
    if n < 3:
        return None
    if not np.allclose(D, D.T, atol=1e-8):
        return None
    if np.any(D < -1e-9):
        return None
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    try:
        w, V = np.linalg.eigh(B)
    except Exception:
        return None
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    if w[0] < 1e-9:
        return None
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
    if np.isnan(corr) or corr < 0.99:
        return None
    return coords

def _build_greedy_tour(d, start=0):
    """Build a greedy nearest-neighbor tour starting from a given node."""
    n = d.shape[0]
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        row = d[current]
        # vectorized nearest among unvisited
        tmp = np.full(n, np.inf)
        tmp[list(unvisited)] = row[list(unvisited)]
        next_node = int(np.argmin(tmp))
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return visited

def _two_opt_improve(tour, D, max_iters=1000, tol=1e-12):
    """Apply 2-opt and Or-Opt local search to improve a tour."""
    n = len(tour)
    if n < 4:
        return list(tour)
    best_tour = list(tour)
    best_len = sum(D[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1

        # 2-opt pass
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d_ = best_tour[j], best_tour[(j + 1) % n]
                old = D[a, b] + D[c, d_]
                new = D[a, c] + D[b, d_]
                if new < old - tol:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_len += new - old
                    improved = True
                    break
            if improved:
                break

        # Or-Opt pass (move 1 node)
        if not improved:
            for i in range(n):
                if i == 0:
                    continue
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[(i + 1) % n]
                for j in range(n):
                    if j == i or j == i - 1 or j == (i + 1) % n:
                        continue
                    x = best_tour[j - 1]
                    y = best_tour[j]
                    old = D[a, b] + D[b, c] + D[x, y]
                    new = D[a, c] + D[x, b] + D[b, y]
                    if new < old - tol:
                        best_tour.pop(i)
                        best_tour.insert(j, b)
                        best_len += new - old
                        improved = True
                        break
                if improved:
                    break

        # Or-Opt pass (move 2 nodes)
        if not improved:
            for i in range(n - 1):
                if i == 0:
                    continue
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[i + 1]
                d_ = best_tour[(i + 2) % n]
                for j in range(n):
                    if j in (i, i + 1, i - 1, (i + 2) % n):
                        continue
                    x = best_tour[j - 1]
                    y = best_tour[j]
                    old = D[a, b] + D[b, c] + D[c, d_] + D[x, y]
                    new = D[a, c] + D[c, b] + D[b, d_] + D[x, y]
                    if new < old - tol:
                        best_tour.pop(i + 1)
                        best_tour.pop(i)
                        best_tour.insert(j, b)
                        best_tour.insert(j, c)
                        best_len += new - old
                        improved = True
                        break
                if improved:
                    break

    return best_tour

def _segments_cross(p1, p2, p3, p4):
    """Check if segment p1-p2 crosses segment p3-p4 using cross products."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _mst_edge_mask(d):
    """Return a boolean matrix marking the edges of the minimum spanning tree."""
    n = d.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    try:
        # build a dense weighted adjacency for the MST (symmetric, zero diagonal)
        w = d.copy()
        np.fill_diagonal(w, 0.0)
        mst = minimum_spanning_tree(csr_matrix(w))
        rows, cols, _ = mst.nonzero()
        for r, c in zip(rows, cols):
            mask[r, c] = True
            mask[c, r] = True
    except Exception:
        return None
    return mask

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).

    Strategy:
    1. Rank-based heuristic (inverse rank of distance).
    2. MST structural prior: MST edges get a high base weight (connectivity,
       anti-fragmentation); non-MST edges keep inverse-distance locality.
    3. Strong Local Search Prior: boost edges from a 2-opt+Or-Opt optimized
       (and finally polished) tour significantly.
    4. Geometric boosts: Delaunay, Convex Hull, KNN (if MDS recovers coords),
       plus a light crossing-edge penalty.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard (early, cheap fallback)
    if _remaining() < 4.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    diag_mask = np.eye(n, dtype=bool)

    # ---- 1. Rank-based heuristic (inverse rank) ----
    flat_d = d.flatten()
    flat_d_masked = flat_d.copy()
    flat_d_masked[diag_mask.flatten()] = np.inf
    ranks = np.argsort(np.argsort(flat_d_masked)) + 1
    ranks = ranks.reshape(n, n)
    heuristic = 1.0 / ranks
    heuristic[diag_mask] = 0.0

    # ---- 2. MST structural prior ----
    # MST edges enforce connectivity / prevent fragmentation; non-MST edges
    # fall back to inverse-distance locality. We blend multiplicatively with
    # the rank base so it reinforces rather than replaces it.
    mst_mask = None
    if _remaining() > 6.0:
        mst_mask = _mst_edge_mask(d)

    if mst_mask is not None:
        # Inverse-distance locality for the non-MST complement
        inv_dist = 1.0 / (d + 1e-9) ** 2
        np.fill_diagonal(inv_dist, 0.0)
        # Smooth surrogate of a min/max-like emphasis: use a soft emphasis so
        # MST edges are strongly favored but the matrix stays smooth.
        mst_weight = 4.0     # multiplier on MST edges (connectivity prior)
        comp_weight = 0.6    # multiplier on the inverse-distance complement
        base = heuristic.copy()
        base = np.where(mst_mask, base * mst_weight, base * comp_weight)
        # For non-MST edges, blend in the inverse-distance locality so geometric
        # locality is preserved where the rank signal is weak.
        blend = 0.5
        base = np.where(mst_mask, base, base * (1 - blend) + inv_dist * blend * 0.0 + base * blend)
        # Simpler, more stable: keep rank base for MST, mix rank+invdist for rest.
        base = np.where(mst_mask, heuristic * mst_weight,
                        heuristic * (1 - blend) + inv_dist * blend)
        heuristic = base
        heuristic[diag_mask] = 0.0

    # ---- 3. Strong Local Search Prior (multi-start greedy + 2-opt/Or-Opt) ----
    best_tour = None
    best_len = np.inf

    if _remaining() > 12.0:
        starts = [0, 1, 2, 3, 4, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]
        # de-dup, keep within [0, n-1]
        seen = set()
        starts = [s for s in starts if 0 <= s < n and not (s in seen or seen.add(s))]
        for start in starts:
            if _remaining() < 6.0:
                break
            greedy_tour = _build_greedy_tour(d, start=start)
            improved_tour = _two_opt_improve(greedy_tour, d, max_iters=500)
            tour_len = sum(d[improved_tour[i], improved_tour[(i + 1) % n]] for i in range(n))
            if tour_len < best_len:
                best_len = tour_len
                best_tour = improved_tour

        # High-precision polish pass on the best tour (tighter tol, more iters)
        if best_tour is not None and _remaining() > 6.0:
            polished = _two_opt_improve(best_tour, d, max_iters=1200, tol=1e-14)
            pol_len = sum(d[polished[i], polished[(i + 1) % n]] for i in range(n))
            if pol_len <= best_len + 1e-12:
                best_tour = polished
                best_len = pol_len

        if best_tour is not None:
            # Boost edges in the best (polished) tour. Slightly lower than before
            # because the MST + geometric priors now share the load, making the
            # signal more robust than a single dominant tour boost.
            tour_boost = 60.0
            for i in range(n):
                u = best_tour[i]
                v = best_tour[(i + 1) % n]
                heuristic[u, v] *= tour_boost
                heuristic[v, u] *= tour_boost

    # ---- 4. Geometric boosts (MDS -> Delaunay / Hull / KNN / crossing penalty) ----
    coords = None
    if _remaining() > 4.0:
        coords = _mds_coordinates(d)

    if coords is not None and _remaining() > 2.0:
        try:
            # KNN boost
            k = min(6, n - 1)
            knn_boost = 2.0
            for i in range(n):
                dists = d[i].copy()
                dists[i] = np.inf
                nn_indices = np.argpartition(dists, k)[:k]
                for j in nn_indices:
                    heuristic[i, j] *= knn_boost
                    heuristic[j, i] *= knn_boost

            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 3.0
            delaunay_edges = set()
            for simplex in tri.simplices:
                i, j, k = simplex
                for (a, b) in ((i, j), (j, k), (i, k)):
                    heuristic[a, b] *= delaunay_boost
                    heuristic[b, a] *= delaunay_boost
                    delaunay_edges.add((min(a, b), max(a, b)))

            # Convex Hull boost
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 10.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost

            # Light crossing-edge penalty using the best tour as a reference
            if best_tour is not None and _remaining() > 1.0:
                tour_edges = set()
                for i in range(n):
                    u = best_tour[i]
                    v = best_tour[(i + 1) % n]
                    tour_edges.add((min(u, v), max(u, v)))
                crossing_penalty = 0.15
                for i in range(n):
                    for j in range(i + 1, n):
                        if (i, j) in tour_edges or (i, j) in delaunay_edges:
                            continue
                        crosses = False
                        for (u, v) in tour_edges:
                            if i in (u, v) or j in (u, v):
                                continue
                            if _segments_cross(coords[i], coords[j], coords[u], coords[v]):
                                crosses = True
                                break
                        if crosses:
                            heuristic[i, j] *= crossing_penalty
                            heuristic[j, i] *= crossing_penalty
        except Exception:
            pass

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)

    # Final safety: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
