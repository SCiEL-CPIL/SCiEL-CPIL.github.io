"""Heuristic for ACO-CVRP: Voronoi-based capacity-aware spatial heuristic."""
import time
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.distance import cdist

_T0 = time.time()
_DEADLINE = _T0 + 300.0
_TIME_MARGIN = 15.0
_N = 51


def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()


def _safe_heuristic(dm):
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    H = 1.0 / (d + 1e-9)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    return H


def _distmat(coords):
    c = np.asarray(coords, dtype=float)
    return np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))


def _voronoi_cell_radius(vor, idx):
    """Estimate the radius of the minimum enclosing circle of a Voronoi cell."""
    try:
        region = vor.regions[vor.point_region[idx]]
        if -1 in region or len(region) < 3:
            # Unbounded or invalid region; use distance to nearest other point
            return 1.0
        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return 1.0
        # Compute centroid and max distance from centroid to vertices as a proxy
        centroid = vertices.mean(axis=0)
        radii = np.sqrt(((vertices - centroid) ** 2).sum(axis=1))
        return float(radii.max())
    except Exception:
        return 1.0


def _build_voronoi_heuristic(dm, coords, demands, capacity, alpha=1.0, S=1.0, dw=0.0):
    """Build heuristic using Voronoi tessellation and capacity-aware penalties."""
    c = np.asarray(coords, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = c.shape[0]
    
    # Compute base inverse-distance heuristic
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    base = 1.0 / (d + 1e-9)
    
    # Compute Voronoi tessellation for customers (nodes 1..n-1)
    # Include depot in the point set for spatial context
    try:
        vor = Voronoi(c)
        cell_radii = np.ones(n)
        for i in range(n):
            cell_radii[i] = _voronoi_cell_radius(vor, i)
    except Exception:
        # Fallback: use k-nearest neighbor distances as proxy for cell size
        cell_radii = np.ones(n)
        try:
            dists = cdist(c, c)
            np.fill_diagonal(dists, np.inf)
            # Use distance to 3rd nearest neighbor as cell size proxy
            nn_dists = np.partition(dists, 2, axis=1)[:, 2]
            cell_radii = nn_dists
        except Exception:
            cell_radii = np.ones(n)
    
    # Normalize cell radii
    cr_min = cell_radii.min()
    cr_max = cell_radii.max()
    if cr_max > cr_min:
        cell_radii_norm = (cell_radii - cr_min) / (cr_max - cr_min)
    else:
        cell_radii_norm = np.ones(n) * 0.5
    
    # Capacity-aware penalty: penalize edges that connect regions with high demand density
    # if the edge is long relative to local cell sizes
    H = S * (base ** alpha)
    
    # Apply Voronoi-based penalty: for each edge (i,j), penalize if distance is large
    # relative to the minimum cell radius of i and j, especially when demands are high
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist_ij = d[i, j]
            min_cr = min(cell_radii[i], cell_radii[j])
            if min_cr < 1e-6:
                min_cr = 1e-6
            ratio = dist_ij / min_cr
            # Penalize long-range connections
            penalty = 1.0 / (1.0 + 0.5 * (ratio - 1.0) ** 2 if ratio > 1.0 else 0.0)
            # Capacity density factor: higher demand in the cell means tighter constraints
            demand_factor = 1.0 + (D[i] + D[j]) / (2.0 * cap)
            H[i, j] *= penalty / demand_factor
    
    # Depot-aware boost
    if dw > 0.0:
        depot_dist = np.sqrt((c - c[0]) ** 2).sum(axis=1)
        depot_boost = 1.0 + dw / (depot_dist + 1e-9)
        H *= depot_boost[None, :]
        H *= depot_boost[:, None]
    
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    return H


def _ant_route(dist, D, rem, cap, logP, logH, max_steps):
    node = 0
    rem = cap
    cost = 0.0
    visited = np.zeros(_N, bool)
    visited[0] = True
    steps = 0
    while True:
        feas = []
        for j in range(1, _N):
            if not visited[j] and D[j] <= rem + 1e-9:
                feas.append(j)
        if not feas:
            cost += dist[node, 0]
            return cost
        vals = np.array([logP[node, j] + logH[node, j] for j in feas])
        m = vals.max()
        w = np.exp(vals - m)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            nxt = feas[0]
        else:
            r = np.random.random() * s
            acc = 0.0
            for k, j in enumerate(feas):
                acc += w[k]
                if r <= acc:
                    nxt = j
                    break
            else:
                nxt = feas[-1]
        cost += dist[node, nxt]
        rem -= D[nxt]
        visited[nxt] = True
        node = nxt
        steps += 1
        if steps >= max_steps:
            cost += dist[node, 0]
            return cost


def _colony(dist, D, cap, logH, n_ants=20, n_iter=30, max_steps=300):
    P = np.ones((_N, _N))
    bests = []
    for _ in range(n_iter):
        itcost = np.empty(n_ants)
        logP = np.log(P)
        for a in range(n_ants):
            c = _ant_route(dist, D, cap, cap, logP, logH, max_steps)
            itcost[a] = c
        b = float(itcost.min())
        bests.append(b)
        P *= 0.9
        P += 1.0 / b
    return min(bests)


def heuristics(distance_matrix, coordinates, demands, capacity):
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    try:
        if dm.shape != (n, n):
            raise ValueError("shape")
        if not np.isfinite(dm).all() or not np.isfinite(coords).all() or not np.isfinite(D).all():
            raise ValueError("non-finite")
        
        if n != _N:
            return _safe_heuristic(dm)
        
        dist = _distmat(coords)
        
        if _remaining() <= 30.0:
            return _build_voronoi_heuristic(dm, coords, D, cap, 1.0, 1.0, 0.0)
        
        # Quick evaluation with a few candidate (alpha, S, dw) tuples
        best_m = 1e30
        best_params = (1.0, 1.0, 0.0)
        
        alpha_vals = [0.7, 1.0, 1.3, 1.6, 2.0]
        S_vals = [0.5, 1.0, 2.0, 4.0]
        dw_vals = [0.0, 0.3, 0.6, 1.0]
        
        n_ants_short = 8
        n_iter_short = 6
        max_steps_short = 120
        
        for alpha in alpha_vals:
            for S in S_vals:
                for dw in dw_vals:
                    if _remaining() <= 30.0:
                        break
                    H = _build_voronoi_heuristic(dm, coords, D, cap, alpha, S, dw)
                    logH = np.log(H)
                    m = _colony(dist, D, cap, logH, n_ants_short, n_iter_short, max_steps_short)
                    if m < best_m:
                        best_m = m
                        best_params = (alpha, S, dw)
                if _remaining() <= 30.0:
                    break
            if _remaining() <= 30.0:
                break
        
        alpha_opt, S_opt, dw_opt = best_params
        
        # Refine around the best with full colony if time permits
        if _remaining() > 50.0:
            cur_a, cur_S, cur_dw = alpha_opt, S_opt, dw_opt
            cur_H = _build_voronoi_heuristic(dm, coords, D, cap, cur_a, cur_S, cur_dw)
            cur_logH = np.log(cur_H)
            cur_m = _colony(dist, D, cap, cur_logH, 15, 12, 200)
            
            for _round in range(5):
                if _remaining() <= 40.0:
                    break
                improved = False
                for da in [0.2, -0.2, 0.1, -0.1]:
                    na = cur_a + da
                    if na < 0.2 or na > 5.0:
                        continue
                    Ht = _build_voronoi_heuristic(dm, coords, D, cap, na, cur_S, cur_dw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_a, cur_H, cur_logH, cur_m = na, Ht, logHt, mt
                        improved = True
                for ds in [1.4, 1/1.4, 1.2, 1/1.2]:
                    nS = cur_S * ds
                    if nS < 0.1 or nS > 20.0:
                        continue
                    Ht = _build_voronoi_heuristic(dm, coords, D, cap, cur_a, nS, cur_dw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_S, cur_H, cur_logH, cur_m = nS, Ht, logHt, mt
                        improved = True
                for ddw in [0.3, -0.3, 0.15, -0.15]:
                    ndw = cur_dw + ddw
                    if ndw < 0.0 or ndw > 3.0:
                        continue
                    Ht = _build_voronoi_heuristic(dm, coords, D, cap, cur_a, cur_S, ndw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_dw, cur_H, cur_logH, cur_m = ndw, Ht, logHt, mt
                        improved = True
                if not improved:
                    break
            
            alpha_opt, S_opt, dw_opt = cur_a, cur_S, cur_dw
        
        H_final = _build_voronoi_heuristic(dm, coords, D, cap, alpha_opt, S_opt, dw_opt)
        return H_final
        
    except Exception:
        try:
            return _safe_heuristic(dm)
        except Exception:
            return np.ones_like(dm)
