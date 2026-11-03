"""Rewritten packing program: Langevin dynamics + SLSQP epigraph + Delaunay sparse polish."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree, Delaunay

DEADLINE = 115.0  # safety margin before the 150s kill


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
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _langevin_dynamics(c0, r, time_budget=8.0, max_iter=200000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r.
    Uses velocity-Verlet with thermal noise to escape local minima.
    Returns a feasible layout for radius r if found."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    
    start = time.time()
    dt = 0.002
    gamma = 0.1  # friction coefficient
    T_init = 0.01
    T_min = 0.00001
    n = max_iter
    
    best_c = c.copy()
    best_score = _true_score(c)
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Temperature schedule
        T = T_init * (T_min / T_init) ** (i / max(1, n))
        
        # Compute forces
        disp = np.zeros_like(c)
        
        # Pairwise overlap repulsion (using KDTree for efficiency)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        
        if len(pairs) > 0:
            for idx in pairs:
                i, j = idx[0], idx[1]
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
        
        # Wall forces
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        
        # Langevin update: velocity-Verlet with noise
        # v_new = (v - dt * force) * exp(-gamma * dt) + sqrt(2 * T * dt * gamma) * noise
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        
        # Position update
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        
        # Check feasibility
        feasible = True
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        if len(pairs_new) > 0:
            feasible = False
        
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                # Accept with Metropolis criterion if close
                if np.random.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            # Reject, keep velocity but damp
            v *= 0.5
    
    return best_c


def _active_set_refine(c0, r, time_budget=5.0):
    """Refine using active-set method: identify tight constraints and solve
    a small QP to maximize minimum distance while maintaining contacts."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    
    for _ in range(10):
        if time.time() - start > time_budget:
            break
        
        # Find tight constraints
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        w = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        
        min_pair = d.min()
        min_wall = w.min()
        
        # Identify active constraints (within tolerance)
        tol = 1e-6
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(i, j) for i, j in active_pairs if i < j]
        
        active_walls = []
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    pen = c[k, axis] - (0.5 - r)
                else:
                    pen = (-0.5 + r) - c[k, axis]
                if pen < tol:
                    active_walls.append((k, axis, sign))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build a simple Newton step: move points to increase min distance
        # For each active pair, push apart along the connecting vector
        disp = np.zeros_like(c)
        for i, j in active_pairs:
            diff = c[i] - c[j]
            d_ij = np.linalg.norm(diff)
            if d_ij > 1e-12:
                force = diff / d_ij * 0.001
                disp[i] += force
                disp[j] -= force
        
        for k, axis, sign in active_walls:
            disp[k, axis] += sign * 0.001
        
        # Apply small step
        c = np.clip(c + 0.01 * disp, -0.5 + r, 0.5 - r)
    
    return c


def _delaunay_sparse_objective(x, k, edges):
    """Sparse soft-min using only Delaunay edges + walls."""
    pts = x.reshape(-1, 2)
    # Pairwise distances for Delaunay edges only
    i_idx, j_idx = edges[:, 0], edges[:, 1]
    diff = pts[i_idx] - pts[j_idx]
    dists = np.sqrt((diff ** 2).sum(-1))
    dists = np.clip(dists, 1e-8, 1.0)
    w_pairs = np.exp(-k * dists)
    
    # Wall distances
    dw = np.stack([
        0.5 - pts[:, 0], 0.5 + pts[:, 0],
        0.5 - pts[:, 1], 0.5 + pts[:, 1],
    ], axis=1)  # shape (N, 4)
    dw = np.clip(dw, 1e-8, 1.0)
    w_walls = np.exp(-k * dw)
    
    all_w = np.concatenate([w_pairs, w_walls.ravel()])
    sw = all_w.sum()
    val = -np.log(sw) / k
    
    # Gradient
    grad_pts = np.zeros_like(pts)
    # Pair contributions
    wp = w_pairs / sw
    safe_d = np.where(dists > 1e-8, dists, 1.0)
    factors = wp / safe_d
    for idx in range(len(edges)):
        i, j = i_idx[idx], j_idx[idx]
        d_vec = diff[idx]
        f = factors[idx]
        grad_pts[i] += f * d_vec
        grad_pts[j] -= f * d_vec
    
    # Wall contributions
    ww = w_walls / sw
    # dw[i,0] = 0.5 - x_i, dw[i,1] = 0.5 + x_i, dw[i,2] = 0.5 - y_i, dw[i,3] = 0.5 + y_i
    grad_pts[:, 0] += ww[:, 1] - ww[:, 0]
    grad_pts[:, 1] += ww[:, 3] - ww[:, 2]
    
    return val, -grad_pts.ravel()


def _delaunay_refine(c0, k, time_budget=3.0):
    """L-BFGS-B polish using Delaunay-sparse objective."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    
    try:
        tri = Delaunay(c0)
        edges = np.unique(tri.simplices, axis=0)
        # Ensure edges are undirected pairs
        edge_set = set()
        unique_edges = []
        for s in tri.simplices:
            for a in range(3):
                i, j = s[a], s[(a + 1) % 3]
                if i > j:
                    i, j = j, i
                if (i, j) not in edge_set:
                    edge_set.add((i, j))
                    unique_edges.append((i, j))
        edges = np.array(unique_edges, dtype=int)
    except Exception:
        # Fallback to all pairs
        edges = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
    
    def obj_and_grad(x):
        return _delaunay_sparse_objective(x, k, edges)
    
    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-15, 'gtol': 1e-10,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _slsqp_epigraph(c0, r_init, time_budget=5.0):
    """SLSQP max-min radius optimization with linear epigraph constraints.
    Variables: x = [c1, c2, ..., cN, t] where t is the radius.
    Maximize t subject to:
      |ci - cj| >= 2t for all i<j
      wall distances >= t
      -0.5 <= ci <= 0.5
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r_init, 0.01)
    x0 = np.concatenate([c0.ravel(), [r0]])
    
    # Constraint functions: g(x) >= 0
    def constraints(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        con = []
        # Pairwise: |ci-cj|^2 >= (2t)^2  =>  ci-cj dot ci-cj - 4t^2 >= 0
        # For SLSQP, use linear approximations around current t
        # Actually use the nonlinear form
        for i in range(N):
            for j in range(i+1, N):
                diff = pts[i] - pts[j]
                con.append(np.dot(diff, diff) - 4*t*t)
        # Walls
        for i in range(N):
            con.append(0.5 - pts[i,0] - t)
            con.append(pts[i,0] + 0.5 - t)
            con.append(0.5 - pts[i,1] - t)
            con.append(pts[i,1] + 0.5 - t)
        return con
    
    def constraint_grad(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        grads = []
        for i in range(N):
            for j in range(i+1, N):
                diff = pts[i] - pts[j]
                g = np.zeros(2*N+1)
                g[2*i] = 2*diff[0]
                g[2*i+1] = 2*diff[1]
                g[2*j] = -2*diff[0]
                g[2*j+1] = -2*diff[1]
                g[2*N] = -8*t
                grads.append(g)
        for i in range(N):
            g = np.zeros(2*N+1)
            g[2*i] = -1; g[2*N] = -1
            grads.append(g)
            g = np.zeros(2*N+1)
            g[2*i] = 1; g[2*N] = -1
            grads.append(g)
            g = np.zeros(2*N+1)
            g[2*i+1] = -1; g[2*N] = -1
            grads.append(g)
            g = np.zeros(2*N+1)
            g[2*i+1] = 1; g[2*N] = -1
            grads.append(g)
        return np.array(grads)
    
    def obj(x):
        return -x[2*N]  # minimize -t = maximize t
    
    def obj_grad(x):
        g = np.zeros(2*N+1)
        g[2*N] = -1
        return g
    
    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)],
                       constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_grad},
                       options={'maxiter': 80, 'ftol': 1e-12, 'disp': False})
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
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
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(10):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 20:
                break
            cur = _force_relax(cur, r, time_budget=1.5, max_iter=3000)
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

    # 2) Langevin dynamics to escape local minima at current best radius
    if time.time() - start_time < DEADLINE - 25:
        ld_budget = min(8.0, DEADLINE - 25 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=80000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
    
    # 3) Active-set refinement on the best layout
    if time.time() - start_time < DEADLINE - 15:
        as_budget = min(4.0, DEADLINE - 15 - (time.time() - start_time))
        if as_budget > 1.0:
            cur = _active_set_refine(best, best_r, time_budget=as_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) SLSQP epigraph polish — direct max-min radius with constraints
    if time.time() - start_time < DEADLINE - 12:
        slsqp_budget = min(8.0, DEADLINE - 12 - (time.time() - start_time))
        if slsqp_budget > 2.0:
            c = _slsqp_epigraph(best, best_r, time_budget=slsqp_budget)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 5) Delaunay-sparse soft-min polish (faster than full O(N^2))
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))
    for c0 in inits[:3]:
        if time.time() - start_time > DEADLINE - 8:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=1500))

    for k in (100.0, 200.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 4:
                break
            c = _delaunay_refine(c0, k, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 6) Final high-precision SLSQP + Delaunay polish
    if time.time() - start_time < DEADLINE - 2:
        c = _slsqp_epigraph(best, best_r, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
        if time.time() - start_time < DEADLINE - 1:
            c = _delaunay_refine(best, 500.0, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
