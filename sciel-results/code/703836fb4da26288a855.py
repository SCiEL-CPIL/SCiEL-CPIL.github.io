"""Heuristic matrix for ACO-TSP: MST-complement linear blend with 2-opt tour prior."""
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

def _remaining():
    return _DEADLINE - time.time()

def _nearest_neighbor_tour(D, start=0):
    """Build a nearest-neighbor tour."""
    n = D.shape[0]
    visited = [start]
    unvisited = list(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        best = min(unvisited, key=lambda x: D[current, x])
        visited.append(best)
        unvisited.remove(best)
        current = best
    return visited

def _two_opt(tour, D, max_passes=400):
    """2-opt local search."""
    n = len(tour)
    if n < 4:
        return list(tour)
    t = list(tour)
    L = sum(D[t[i], t[(i+1) % n]] for i in range(n))
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = t[i], t[i+1]
                c, d = t[j], t[(j+1) % n]
                delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
                if delta < -1e-12:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    L += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return t

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Base: inverse distance (power 1) for natural locality.
    2. MST edges get a strong multiplicative boost to enforce connectivity.
    3. Non-MST edges are damped to prevent fragmentation.
    4. Best 2-opt tour edges get an additional boost.
    5. Diagonal zeroed; non-positive/non-finite replaced with 1e-9.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        h = 1.0 / (d + 1e-10)
        np.fill_diagonal(h, 0.0)
        return np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    if _remaining() < 5.0:
        h = 1.0 / (d + 1e-10)
        np.fill_diagonal(h, 0.0)
        return np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Base heuristic: simple inverse distance
    h = 1.0 / (d + 1e-10)
    
    # MST boost: strongly favor MST edges for connectivity
    if _remaining() > 5.0:
        try:
            D_mst = d.copy()
            np.fill_diagonal(D_mst, np.inf)
            mst = minimum_spanning_tree(csr_matrix(D_mst))
            mst_dense = mst.toarray()
            # MST edges have positive weight; use a moderate boost
            mst_mask = (mst_dense > 0)
            h = h * np.where(mst_mask, 6.0, 1.0)
            # Symmetrize
            h = (h + h.T) / 2.0
        except Exception:
            pass
    
    # Tour prior from multi-start 2-opt
    if _remaining() > 5.0:
        best_tour = None
        best_len = np.inf
        for start in [0, n // 3, 2 * n // 3]:
            if _remaining() < 3.0:
                break
            nn_tour = _nearest_neighbor_tour(d, start=start)
            opt_tour = _two_opt(nn_tour, d, max_passes=200)
            tlen = sum(d[opt_tour[i], opt_tour[(i+1) % n]] for i in range(n))
            if tlen < best_len:
                best_len = tlen
                best_tour = opt_tour
        
        if best_tour is not None:
            # Boost tour edges moderately
            for i in range(n):
                u = best_tour[i]
                v = best_tour[(i + 1) % n]
                h[u, v] *= 8.0
                h[v, u] *= 8.0
    
    # Diagonal zero
    np.fill_diagonal(h, 0.0)
    
    # Sanitize
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
