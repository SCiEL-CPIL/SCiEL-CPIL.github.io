"""Packing 300 equal circles in a unit circle via Delaunay stress relaxation + L-BFGS-B soft-min gradient ascent."""
import numpy as np
import time
import math
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to unit disk, keep n closest to center."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * math.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        spacing = 0.08
        k = int(1.2 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * math.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= 1.0
        pts = pts[inside]
        r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _compute_radius(pts):
    """Exact achieved common radius (O(n^2), double precision)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = math.sqrt(max(np.min(d2), 0.0)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _delaunay_stress_layout(pts, target_r, time_budget, dt=0.002, max_iter=200000):
    """
    Force-directed relaxation on the Delaunay graph.
    - Repulsion: push apart pairs closer than 2*target_r (overlap^4 Hertzian-like).
    - Attraction: pull apart pairs farther than 2*target_r toward it (uniform spacing).
    - Boundary: clamp inside radius (1 - target_r).
    Rebuild Delaunay periodically.
    Returns improved pts.
    """
    n = len(pts)
    pts = pts.copy()
    min_d = 2.0 * target_r
    t0 = time.time()
    rebuild_every = 200
    it = 0
    prev_energy = np.inf

    while True:
        if time.time() - t0 > time_budget:
            break
        if it >= max_iter:
            break

        # Build Delaunay (and its edges) every `rebuild_every` iterations
        if it % rebuild_every == 0:
            try:
                tri = Delaunay(pts)
                # Unique edges from simplices
                simp = tri.simplices
                edges = set()
                for s in simp:
                    edges.add(tuple(sorted((s[0], s[1]))))
                    edges.add(tuple(sorted((s[0], s[2]))))
                    edges.add(tuple(sorted((s[1], s[2]))))
                edges = list(edges)
            except Exception:
                # Fallback: use previous edges or a few random pairs
                if 'edges' not in dir() or edges is None:
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if np.random.random() < 0.01]
                else:
                    pass

        forces = np.zeros_like(pts)
        energy = 0.0

        for (i, j) in edges:
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d2 = dx * dx + dy * dy
            d = math.sqrt(max(d2, 1e-12))
            if d < 1e-9:
                # degenerate: nudge apart randomly
                ang = np.random.uniform(0, 2 * math.pi)
                fx, fy = math.cos(ang), math.sin(ang)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                continue
            # Hertzian overlap^4 repulsion when overlapping
            if d < min_d:
                overlap = (min_d - d) / min_d
                f = overlap ** 4 * 10.0
                ux, uy = dx / d, dy / d
                forces[i, 0] += f * ux
                forces[i, 1] += f * uy
                forces[j, 0] -= f * ux
                forces[j, 1] -= f * uy
                energy += overlap ** 4
            else:
                # mild attraction to bring far pairs toward min_d (uniformity)
                gap = (d - min_d) / min_d
                if gap < 1.0:
                    f = gap * 0.05
                    ux, uy = dx / d, dy / d
                    forces[i, 0] -= f * ux
                    forces[i, 1] -= f * uy
                    forces[j, 0] += f * ux
                    forces[j, 1] += f * uy

        # Integrate
        pts += dt * forces

        # Boundary clamp
        rad = np.hypot(pts[:, 0], pts[:, 1])
        max_allowed = 1.0 - target_r
        out = rad > max_allowed
        if np.any(out):
            pts[out, 0] *= max_allowed / rad[out]
            pts[out, 1] *= max_allowed / rad[out]

        # Damping / stop if converged
        it += 1
        if abs(prev_energy - energy) < 1e-10 and it > 1000:
            break
        prev_energy = energy

    return pts


def _soft_min_value(pts, alpha):
    """
    Smooth min of the constraints (pair distances and boundary margins).
    Returns (value, logsumexp_components).
    value = -logsumexp(-constraints/alpha) * alpha  (approximates min)
    """
    n = len(pts)
    # Pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    d_pairs = d[iu]
    # Boundary margins: 1 - |c_i|
    rad = np.hypot(pts[:, 0], pts[:, 1])
    margins = 1.0 - rad
    # Combine all constraints
    cons = np.concatenate([d_pairs, margins])
    x = -cons / alpha
    m = np.max(x)
    if not np.isfinite(m):
        return 0.0, cons, 0.0
    lse = m + math.log(np.sum(np.exp(x - m)))
    value = -lse * alpha
    return value, cons, lse


def _objective_and_grad(x, alpha, target_r):
    """
    Objective to MAXIMIZE: soft-min of constraints, minus penalty if below target_r.
    We minimize F = -softmin + penalty.
    Returns (F, grad).
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(n, k=1)
    d_pairs = d[iu]
    rad = np.hypot(pts[:, 0], pts[:, 1])
    margins = 1.0 - rad
    cons = np.concatenate([d_pairs, margins])

    xneg = -cons / alpha
    m = np.max(xneg)
    if not np.isfinite(m):
        return 1e6, np.zeros_like(x)
    exps = np.exp(xneg - m)
    s = np.sum(exps)
    # softmin = -alpha * (m + log(s))
    softmin = -alpha * (m + math.log(s))

    # Gradient of softmin w.r.t. cons: p_k = exp(xneg_k)/s
    p = exps / s
    # d softmin / d cons_k = p_k  (since softmin = -alpha*lse, d lse/d cons = -p_k/alpha... let's derive)
    # lse = m + log(sum exp(xneg - m)), xneg = -cons/alpha
    # d lse / d cons_k = (1/alpha) * p_k  ... check: d lse/d xneg_k = p_k; d xneg_k/d cons_k = -1/alpha
    # so d lse/d cons_k = -p_k/alpha
    # softmin = -alpha * lse -> d softmin/d cons_k = -alpha * (-p_k/alpha) = p_k
    # Good: grad_softmin_wrt_cons = p
    p_pairs = p[:len(d_pairs)]
    p_margins = p[len(d_pairs):]

    # d cons / d pts
    # For pair (i,j): d_ij = |c_i - c_j|, d d_ij/d c_i = (c_i - c_j)/d_ij
    grad_pts = np.zeros_like(pts)
    # Accumulate pair contributions
    # d_pairs is in the same order as iu
    for idx in range(len(d_pairs)):
        i, j = iu[0][idx], iu[1][idx]
        w = p_pairs[idx]
        if d_pairs[idx] > 1e-12:
            ux = (pts[i, 0] - pts[j, 0]) / d_pairs[idx]
            uy = (pts[i, 1] - pts[j, 1]) / d_pairs[idx]
            grad_pts[i, 0] += w * ux
            grad_pts[i, 1] += w * uy
            grad_pts[j, 0] -= w * ux
            grad_pts[j, 1] -= w * uy

    # Boundary margin: m_i = 1 - |c_i|, d m_i/d c_i = -c_i/|c_i|
    for i in range(n):
        w = p_margins[i]
        if rad[i] > 1e-12:
            grad_pts[i, 0] -= w * pts[i, 0] / rad[i]
            grad_pts[i, 1] -= w * pts[i, 1] / rad[i]

    # We maximize softmin, so minimize -softmin. grad of -softmin = -grad_softmin
    F = -softmin
    # Add penalty if softmin < target_r (encourage reaching target)
    if softmin < target_r:
        F += (target_r - softmin) ** 2 * 10.0
        # gradient of penalty: 2*(target_r - softmin)*10 * (-d softmin/d pts) = -20*(target_r-softmin)*grad_softmin
        # but we already have grad_softmin = grad_pts (since d softmin/d pts = grad_pts above)
        # Actually grad_pts above IS d softmin/d pts. So penalty grad = -20*(target_r-softmin)*grad_pts
        F_grad = -grad_pts - 20.0 * (target_r - softmin) * grad_pts
    else:
        F_grad = -grad_pts

    return F, F_grad.reshape(-1)


def _objective_only(x, alpha, target_r):
    F, _ = _objective_and_grad(x, alpha, target_r)
    return F


def _gradient_ascent(pts, target_r, time_budget, alpha=0.005):
    """
    L-BFGS-B on the smooth objective to push the soft-min (and thus the true min) upward.
    """
    x0 = pts.reshape(-1).copy()
    t0 = time.time()

    # Boundaries: each coordinate in [-1, 1] (we also clamp in objective)
    bounds = [(-1.0, 1.0)] * (2 * len(pts))

    def _obj(x):
        return _objective_only(x, alpha, target_r)

    def _grad(x):
        _, g = _objective_and_grad(x, alpha, target_r)
        return g

    # Use a callback to stop on time
    state = {'stop': False}

    def _callback(xk):
        if time.time() - t0 > time_budget - 2:
            state['stop'] = True
            return True
        return False

    try:
        res = minimize(_obj, x0, jac=_grad, method='L-BFGS-B',
                       bounds=bounds, callback=_callback,
                       options={'maxiter': 200, 'maxfun': 50000, 'gtol': 1e-8, 'ftol': 1e-12})
        pts_new = res.x.reshape(-1, 2)
    except Exception:
        pts_new = pts.copy()

    # Clamp to be safe
    rad = np.hypot(pts_new[:, 0], pts_new[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts_new[out, 0] /= rad[out]
        pts_new[out, 1] /= rad[out]

    return pts_new


def construct():
    start_time = time.time()
    np.random.seed(12345)

    best_pts = _hex_init(N)
    best_score = _compute_radius(best_pts)

    # Multiple restarts with diverse inits
    n_restarts = 3
    time_per_restart = (TIME_LIMIT - (time.time() - start_time)) / (n_restarts + 0.5)

    for rs in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break

        # Diverse init: hex + noise, or rotated hex
        if rs == 0:
            pts = _hex_init(N)
        elif rs == 1:
            pts = _hex_init(N)
            ang = np.random.uniform(0, 2 * math.pi)
            c, s = math.cos(ang), math.sin(ang)
            rot = np.array([[c, -s], [s, c]])
            pts = pts @ rot.T
        else:
            pts = _hex_init(N)
            noise = np.random.normal(0, 0.005, size=(N, 2))
            pts += noise
            rad = np.hypot(pts[:, 0], pts[:, 1])
            out = rad > 1.0
            if np.any(out):
                pts[out, 0] /= rad[out]
                pts[out, 1] /= rad[out]

        cur_score = _compute_radius(pts)

        # Phase A: Delaunay stress relaxation at a slightly optimistic target
        target_r = min(cur_score + 0.003, 0.055)
        stress_time = min(time_per_restart * 0.45, 40.0)
        pts = _delaunay_stress_layout(pts, target_r, stress_time)
        cur_score = _compute_radius(pts)

        # Phase B: L-BFGS-B gradient ascent on soft-min, ramping target up
        remain = TIME_LIMIT - (time.time() - start_time) - 5.0
        if remain > 10:
            # Try to push toward a higher target
            target2 = min(cur_score + 0.002, 0.056)
            grad_time = min(remain * 0.7, 45.0)
            pts = _gradient_ascent(pts, target2, grad_time, alpha=0.004)
            cur_score = _compute_radius(pts)

        # Phase C: short final gradient polish at the achieved radius to nudge up
        remain = TIME_LIMIT - (time.time() - start_time) - 3.0
        if remain > 5:
            target3 = min(cur_score + 0.0005, 0.056)
            pts = _gradient_ascent(pts, target3, min(remain, 15.0), alpha=0.002)
            cur_score = _compute_radius(pts)

        if cur_score > best_score:
            best_score = cur_score
            best_pts = pts.copy()

    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out, 0] /= rad[out]
        best_pts[out, 1] /= rad[out]

    return best_pts
