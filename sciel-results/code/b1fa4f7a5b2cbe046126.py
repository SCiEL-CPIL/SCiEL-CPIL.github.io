import numpy as np
import time
from scipy.spatial import Delaunay

_DEADLINE = 320.0  # safety margin under 330s wall-clock kill

def _elapsed(start):
    return time.time() - start

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = d[cur]
        row = row.copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return tour

def _two_opt(tour, d):
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if improved:
                break
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                if j == n - 1:
                    c, dd = tour[j], tour[0]
                else:
                    c, dd = tour[j], tour[j + 1]
                delta = d[a, c] + d[b, dd] - d[a, b] - d[c, dd]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    break
    return tour

def _build_geometric_prior(d, pts):
    """
    Geometry-aware edge attractance using Delaunay triangulation.
    For each Delaunay edge, score = local_scale / edge_length,
    where local_scale comes from the circumradius of adjacent triangles.
    Returns a (n,n) matrix of non-negative geometric scores.
    """
    n = d.shape[0]
    geo = np.zeros((n, n))
    try:
        tri = Delaunay(pts)
        # For each edge, accumulate circumradius of triangles containing it
        edge_circ = {}
        for simplex in tri.simplices:
            a, b, c = simplex
            ax, ay = pts[a]
            bx, by = pts[b]
            cx, cy = pts[c]
            # circumradius = abc / (4 * area)
            d_ab = np.hypot(ax - bx, ay - by)
            d_bc = np.hypot(bx - cx, by - cy)
            d_ca = np.hypot(cx - ax, cy - ay)
            # area via cross product
            area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
            area = 0.5 * area2
            if area < 1e-15:
                R = np.inf
            else:
                R = (d_ab * d_bc * d_ca) / (4.0 * area)
            for (u, v) in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edge_circ.setdefault(key, []).append(R)
        # For each Delaunay edge, use the min circumradius (tightest local fit)
        for (u, v), Rs in edge_circ.items():
            local_scale = min(Rs)
            L = d[u, v]
            if L < 1e-15:
                score = 1.0
            else:
                # reward short edges relative to local scale; normalize
                score = local_scale / (L + 1e-12)
            # cap to avoid blowups
            score = min(score, 50.0)
            geo[u, v] = score
            geo[v, u] = score
    except Exception:
        # fallback: pure inverse-distance geometric score
        with np.errstate(divide='ignore'):
            geo = 1.0 / (d + 1e-9)
    return geo

def _simulate_aco(d, log_h, n_ants=20, n_iter=30, seed=42):
    """
    Fast ACO. log_h = log of (sanitized) heuristic, shape (n,n).
    Pheromone starts at 1. Returns best closed-tour length found.
    """
    n = d.shape[0]
    rng = np.random.RandomState(seed)
    tau = np.ones((n, n))
    log_tau = np.zeros((n, n))
    best_len = np.inf

    for it in range(n_iter):
        deposits = np.zeros((n, n))
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            cur = 0
            tour = [0]
            length = 0.0
            for _ in range(n - 1):
                # attractiveness = tau * h  -> log = log_tau + log_h
                log_attr = log_tau[cur, :] + log_h[cur, :]
                log_attr[visited] = -np.inf
                # softmax via log-sum-exp for numerical stability
                m = np.max(log_attr)
                if not np.isfinite(m):
                    unv = np.where(~visited)[0]
                    nxt = int(rng.choice(unv))
                else:
                    e = np.exp(log_attr - m)
                    s = np.sum(e)
                    if s <= 0.0:
                        unv = np.where(~visited)[0]
                        nxt = int(rng.choice(unv))
                    else:
                        p = e / s
                        nxt = int(rng.choice(n, p=p))
                length += d[cur, nxt]
                tour.append(nxt)
                visited[nxt] = True
                cur = nxt
            length += d[cur, 0]
            tour.append(0)
            if length < best_len:
                best_len = length
            dep = 1.0 / length
            for k in range(len(tour) - 1):
                deposits[tour[k], tour[k + 1]] += dep
        tau = tau * 0.9 + deposits
        log_tau = np.log(tau + 1e-300)
    return best_len

def heuristics(distance_matrix):
    start = time.time()
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance matrix must be square")
    n = d.shape[0]
    d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)

    # Recover points (approx) for Delaunay: use row-wise embedding via
    # classical MDS on the distance matrix to get 2-D coordinates.
    from scipy.spatial.distance import squareform
    try:
        D = d.copy()
        # ensure symmetric
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        # classical MDS
        D2 = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        evals, evecs = np.linalg.eigh(B)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        evals = np.clip(evals, 0.0, None)
        pts = evecs[:, :2] * np.sqrt(evals[:2])
    except Exception:
        # fallback: use random points in unit square seeded by n
        rng = np.random.RandomState(0)
        pts = rng.rand(n, 2)

    geo = _build_geometric_prior(d, pts)

    # Base inverse distance
    with np.errstate(divide='ignore'):
        inv_dist = 1.0 / (d + 1e-9)

    # Normalize geo to [0,1]-ish scale for blending
    gmax = np.max(geo)
    if gmax > 0:
        geo_norm = geo / gmax
    else:
        geo_norm = geo

    # Candidate heuristic builder
    def build_h(alpha, gamma, sharp):
        # base = inv_dist ** alpha
        base = np.power(inv_dist, alpha)
        # geometric boost: (1 + gamma * geo_norm ** sharp)
        boost = 1.0 + gamma * np.power(geo_norm, sharp)
        h = base * boost
        return h

    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h[~np.isfinite(h)] = 1e-9
        h[h <= 0] = 1e-9
        return h

    def score_heuristic(h, seeds):
        hs = sanitize(h)
        log_h = np.log(hs)
        total = 0.0
        for s in seeds:
            total += _simulate_aco(d, log_h, seed=s)
        return total / len(seeds)

    # Search space
    alphas = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    gammas = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    sharps = np.array([0.5, 1.0, 1.5, 2.0])

    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    best_score = np.inf
    best_h = None
    best_params = None

    # Coarse grid
    for alpha in alphas:
        if _elapsed(start) > _DEADLINE:
            break
        for gamma in gammas:
            if _elapsed(start) > _DEADLINE:
                break
            for sharp in sharps:
                if _elapsed(start) > _DEADLINE:
                    break
                h = build_h(alpha, gamma, sharp)
                sc = score_heuristic(h, seeds)
                if sc < best_score:
                    best_score = sc
                    best_h = h.copy()
                    best_params = (alpha, gamma, sharp)

    # Local refinement around best
    if best_params is not None and _elapsed(start) < _DEADLINE - 40:
        ba, bg, bs = best_params
        for alpha in np.linspace(max(0.5, ba - 0.75), ba + 0.75, 7):
            if _elapsed(start) > _DEADLINE - 20:
                break
            for gamma in np.linspace(max(0.0, bg - 1.5), bg + 1.5, 7):
                if _elapsed(start) > _DEADLINE - 20:
                    break
                for sharp in np.linspace(max(0.25, bs - 0.75), bs + 0.75, 5):
                    if _elapsed(start) > _DEADLINE - 20:
                        break
                    h = build_h(alpha, gamma, sharp)
                    sc = score_heuristic(h, seeds)
                    if sc < best_score:
                        best_score = sc
                        best_h = h.copy()
                        best_params = (alpha, gamma, sharp)

    if best_h is None:
        best_h = sanitize(np.power(inv_dist, 2.0))

    return best_h
