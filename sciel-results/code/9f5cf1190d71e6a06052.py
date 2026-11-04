"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.optimize import LinearConstraint

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Hard-disk relaxation with adaptive step size and damping."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.03
    prev_overlap = np.inf
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        # pairwise overlap repulsion
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        # adaptive damping: reduce step if overlap isn't decreasing
        total_overlap = np.sum(np.clip(2*r - d, 0, None))
        if total_overlap > prev_overlap * 1.05:
            dt *= 0.7
        else:
            dt = min(dt * 1.02, 0.08)
        prev_overlap = total_overlap
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _softmin_refine(c0, k, time_budget=3.0, maxiter=200, maxfun=5000,
                    ftol=1e-15, gtol=1e-10):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': maxiter, 'maxfun': maxfun,
                                'ftol': ftol, 'gtol': gtol,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _delaunay_contact_graph(c, r):
    """Return set of (i,j) pairs that are near-contact in the Delaunay graph."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    edges.add((min(i, j), max(i, j)))
        # Filter to near-contact edges (within 2.3*r to capture potential contacts)
        near = set()
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.3 * r:
                near.add((i, j))
        return near
    except Exception:
        # Fallback: all pairs within 2.5*r
        near = set()
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2.5 * r:
                    near.add((i, j))
        return near


def _slsqp_epigraph(c0, r0, time_budget=5.0, maxiter=200, ftol=1e-14):
    """SLSQP epigraph formulation: maximize t subject to
    t <= |c_i - c_j|/2 for all near-contact pairs,
    t <= wall distances for all points.
    Uses vectorized constraints from Delaunay contact graph."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)
    
    # Get contact graph - slightly wider to catch near-binding constraints
    contacts = _delaunay_contact_graph(c0, r0)
    contact_list = list(contacts)
    n_contacts = len(contact_list)
    
    def obj(x):
        return -x[-1]  # minimize -t = maximize t
    
    def obj_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g
    
    def constraint_fun(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[-1]
        pair_cons = np.zeros(n_contacts)
        for idx, (i, j) in enumerate(contact_list):
            diff = pts[i] - pts[j]
            d = np.sqrt(diff[0]**2 + diff[1]**2)
            pair_cons[idx] = d / 2.0 - t
        wall_cons = np.zeros(4 * N)
        wall_cons[0::4] = pts[:, 0] + 0.5 - t
        wall_cons[1::4] = 0.5 - pts[:, 0] - t
        wall_cons[2::4] = pts[:, 1] + 0.5 - t
        wall_cons[3::4] = 0.5 - pts[:, 1] - t
        return np.concatenate([pair_cons, wall_cons])
    
    def constraint_jac(x):
        pts = x[:2*N].reshape(N, 2)
        n_total = n_contacts + 4 * N
        J = np.zeros((n_total, 2 * N + 1))
        
        for idx, (i, j) in enumerate(contact_list):
            diff = pts[i] - pts[j]
            d = np.sqrt(diff[0]**2 + diff[1]**2)
            if d < 1e-12:
                continue
            u0 = diff[0] / d
            u1 = diff[1] / d
            J[idx, 2*i] = 0.5 * u0
            J[idx, 2*i+1] = 0.5 * u1
            J[idx, 2*j] = -0.5 * u0
            J[idx, 2*j+1] = -0.5 * u1
            J[idx, -1] = -1.0
        
        base = n_contacts
        for i in range(N):
            J[base + 4*i, 2*i] = 1.0
            J[base + 4*i, -1] = -1.0
            J[base + 4*i+1, 2*i] = -1.0
            J[base + 4*i+1, -1] = -1.0
            J[base + 4*i+2, 2*i+1] = 1.0
            J[base + 4*i+2, -1] = -1.0
            J[base + 4*i+3, 2*i+1] = -1.0
            J[base + 4*i+3, -1] = -1.0
        
        return J
    
    x0 = np.concatenate([c0.ravel(), [r0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    cons = {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
    
    try:
        start = time.time()
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0})
        if time.time() - start > time_budget:
            return c0
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
        return c
    except Exception:
        return c0


def _adaptive_softmin(c0, k_schedule, time_budget=10.0):
    """Progressively tighten the soft-min approximation via adaptive k schedule.
    Starts with smooth (low k) for global shape, ends with sharp (high k) for precision."""
    c = c0.copy()
    per_step = time_budget / len(k_schedule)
    for k in k_schedule:
        remaining = per_step
        c = _softmin_refine(c, k, time_budget=remaining, maxiter=150, maxfun=4000,
                           ftol=1e-14, gtol=1e-10)
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 12:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=4000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.05
                hi = r
            else:
                hi = r
                if hi - lo < 1e-6:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Adaptive soft-min polish: progressive k tightening from smooth to sharp
    if time.time() - start_time < DEADLINE - 25:
        c = _adaptive_softmin(best, [60.0, 120.0, 250.0, 500.0], time_budget=12.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 3) Primary SLSQP epigraph with generous budget
    if time.time() - start_time < DEADLINE - 20:
        c = _slsqp_epigraph(best, best_r, time_budget=15.0, maxiter=400, ftol=1e-15)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 4) Adaptive soft-min intermediate polish to escape SLSQP local minima
    if time.time() - start_time < DEADLINE - 12:
        c = _adaptive_softmin(best, [150.0, 300.0, 600.0], time_budget=6.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Second SLSQP epigraph pass with tighter tolerance
    if time.time() - start_time < DEADLINE - 6:
        c = _slsqp_epigraph(best, best_r, time_budget=5.0, maxiter=500, ftol=1e-16)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Final high-precision adaptive soft-min polish
    if time.time() - start_time < DEADLINE - 2:
        c = _adaptive_softmin(best, [400.0, 800.0, 1500.0], time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
