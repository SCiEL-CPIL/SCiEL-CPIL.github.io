"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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
            # unit vectors from j to i
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


def _slsqp_epigraph(c0, t0, time_budget=20.0):
    """SLSQP with epigraph variable t: maximize t subject to
    all pairwise distances >= 2t and all wall distances >= t."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    t0 = max(t0, 0.01)
    
    # Precompute pair indices
    ii, jj = np.triu_indices(N, 1)
    n_pairs = len(ii)
    
    def neg_t(x):
        """Minimize -t (equivalent to maximize t)."""
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        return -t
    
    def neg_t_grad(x):
        g = np.zeros(2*N+1)
        g[2*N] = -1.0
        return g
    
    def constraint_fn(x):
        """Return array of constraint values (all must be >= 0)."""
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        # Pairwise: dist(i,j) - 2t >= 0
        diff = pts[ii] - pts[jj]
        dists = np.sqrt((diff**2).sum(axis=1))
        c_pairs = dists - 2.0*t
        # Wall: 0.5 - |c_ik| - t >= 0 for each point and each side
        walls = np.stack([
            0.5 - pts[:,0] - t,   # right
            0.5 + pts[:,0] - t,   # left
            0.5 - pts[:,1] - t,   # top
            0.5 + pts[:,1] - t,   # bottom
        ], axis=1).ravel()
        return np.concatenate([c_pairs, walls])
    
    def constraint_jac(x):
        """Jacobian of constraints w.r.t. x (2N+1 vars)."""
        pts = x[:2*N].reshape(N, 2)
        t = x[2*N]
        # Pairwise gradients
        diff = pts[ii] - pts[jj]
        dists = np.sqrt((diff**2).sum(axis=1))
        safe = np.where(dists > 1e-12, dists, 1e-12)
        # d(dist_ij)/d(pts_i) = (pts_i - pts_j)/dist, d/d(pts_j) = -(pts_i-pts_j)/dist
        # d/d(t) = -2
        jac_pairs = np.zeros((n_pairs, 2*N+1))
        # For each pair (i,j): grad w.r.t. pts_i is diff/dist, pts_j is -diff/dist, t is -2
        jac_pairs[:, ii*2]     += diff[:, 0] / safe
        jac_pairs[:, ii*2+1]   += diff[:, 1] / safe
        jac_pairs[:, jj*2]     -= diff[:, 0] / safe
        jac_pairs[:, jj*2+1]   -= diff[:, 1] / safe
        jac_pairs[:, 2*N]      = -2.0
        
        # Wall gradients: for point k, wall constraint (0.5 - x_k - t)
        # grad w.r.t. x_k = -1, y_k = 0, t = -1
        jac_walls = np.zeros((4*N, 2*N+1))
        # right wall: 0.5 - pts[:,0] - t
        jac_walls[:N, np.arange(N)*2] = -1.0
        jac_walls[:N, 2*N] = -1.0
        # left wall: 0.5 + pts[:,0] - t
        jac_walls[N:2*N, np.arange(N)*2] = 1.0
        jac_walls[N:2*N, 2*N] = -1.0
        # top wall: 0.5 - pts[:,1] - t
        jac_walls[2*N:3*N, np.arange(N)*2+1] = -1.0
        jac_walls[2*N:3*N, 2*N] = -1.0
        # bottom wall: 0.5 + pts[:,1] - t
        jac_walls[3*N:4*N, np.arange(N)*2+1] = 1.0
        jac_walls[3*N:4*N, 2*N] = -1.0
        
        return np.vstack([jac_pairs, jac_walls])
    
    # SLSQP expects constraints as dict or array of dicts
    cons = [{'type': 'ineq', 'fun': constraint_fn, 'jac': constraint_jac}]
    
    x0 = np.concatenate([c0.ravel(), [t0]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
    
    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-15,
                                'disp': 0})
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    
    # Check time
    if time.time() - start > time_budget:
        pass  # SLSQP already returned
    
    return c


def _softmin_refine(c0, k, time_budget=3.0):
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
                       options={'maxiter': 200, 'maxfun': 5000,
                                'ftol': 1e-15, 'gtol': 1e-10,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _contact_graph_refine(c0, time_budget=5.0):
    """Local refinement using only contact (near-minimum) constraints.
    Identify the set of active constraints (pairs and walls at min distance)
    and optimize only those, which is much faster."""
    c = np.array(c0, dtype=float, copy=True)
    r0 = _true_score(c)
    if r0 <= 0:
        return c
    
    start = time.time()
    # Iteratively identify active set and refine
    for _ in range(5):
        if time.time() - start > time_budget:
            break
        r = _true_score(c)
        if r <= 0:
            break
        # Find all constraints within 1.5% of minimum
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        wall_dists = np.stack([
            0.5 - c[:,0], 0.5 + c[:,0],
            0.5 - c[:,1], 0.5 + c[:,1],
        ], axis=1)
        min_pair = np.min(dists)
        min_wall = np.min(wall_dists)
        min_r = min(min_pair/2, min_wall)
        tol = 0.02 * min_r + 1e-6
        
        # Active pairs
        active_pairs = dists < min_pair + 2*tol
        np.fill_diagonal(active_pairs, False)
        ii, jj = np.where(active_pairs)
        
        # Active walls
        active_walls = wall_dists < min_wall + tol
        # active_walls is (N, 4)
        
        if len(ii) == 0:
            break
        
        # Use SLSQP on active constraints only
        n_act = len(ii) + int(active_walls.sum())
        if n_act < 2:
            break
        
        def neg_t(x):
            return -x[2*N]
        
        def neg_t_grad(x):
            g = np.zeros(2*N+1)
            g[2*N] = -1.0
            return g
        
        def active_constraints(x):
            pts = x[:2*N].reshape(N, 2)
            t = x[2*N]
            cons = []
            if len(ii) > 0:
                d = pts[ii] - pts[jj]
                ds = np.sqrt((d**2).sum(axis=1))
                cons.append(ds - 2.0*t)
            if active_walls.any():
                aw = np.where(active_walls)
                cons.append(wall_dists_from_pts(pts, aw) - t)
            return np.concatenate(cons)
        
        def wall_dists_from_pts(pts, aw):
            # aw is tuple of arrays (rows, cols) into (N,4)
            rows, cols = aw
            vals = np.zeros(len(rows))
            for k in range(len(rows)):
                i, side = rows[k], cols[k]
                if side == 0: vals[k] = 0.5 - pts[i,0]
                elif side == 1: vals[k] = 0.5 + pts[i,0]
                elif side == 2: vals[k] = 0.5 - pts[i,1]
                else: vals[k] = 0.5 + pts[i,1]
            return vals
        
        def active_jac(x):
            pts = x[:2*N].reshape(N, 2)
            t = x[2*N]
            jac_list = []
            if len(ii) > 0:
                d = pts[ii] - pts[jj]
                ds = np.sqrt((d**2).sum(axis=1))
                safe = np.where(ds > 1e-12, ds, 1e-12)
                jp = np.zeros((len(ii), 2*N+1))
                jp[:, ii*2] = d[:,0]/safe
                jp[:, ii*2+1] = d[:,1]/safe
                jp[:, jj*2] = -d[:,0]/safe
                jp[:, jj*2+1] = -d[:,1]/safe
                jp[:, 2*N] = -2.0
                jac_list.append(jp)
            if active_walls.any():
                rows, cols = np.where(active_walls)
                n_w = len(rows)
                jw = np.zeros((n_w, 2*N+1))
                for k in range(n_w):
                    i, side = rows[k], cols[k]
                    jw[k, 2*N] = -1.0
                    if side == 0: jw[k, i*2] = -1.0
                    elif side == 1: jw[k, i*2] = 1.0
                    elif side == 2: jw[k, i*2+1] = -1.0
                    else: jw[k, i*2+1] = 1.0
                jac_list.append(jw)
            return np.vstack(jac_list)
        
        cons = [{'type': 'ineq', 'fun': active_constraints, 'jac': active_jac}]
        x0 = np.concatenate([c.ravel(), [r]])
        bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
        
        try:
            res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 100, 'ftol': 1e-14, 'disp': 0})
            c_new = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
            s_new = _true_score(c_new)
            if s_new > _true_score(c) - 1e-9:
                c = c_new
            else:
                break
        except Exception:
            break
    
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
        if time.time() - start_time > DEADLINE - 40:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 40:
                break
            cur = _force_relax(cur, r, time_budget=1.0, max_iter=2000)
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

    # 2) SLSQP epigraph: direct max-min radius optimization (primary optimizer)
    if time.time() - start_time < DEADLINE - 30:
        slsqp_budget = min(25.0, DEADLINE - 30 - (time.time() - start_time))
        if slsqp_budget > 5.0:
            c = _slsqp_epigraph(best, best_r, time_budget=slsqp_budget)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
    
    # 3) Contact-graph local refinement (fast, exploits active constraints)
    if time.time() - start_time < DEADLINE - 15:
        cg_budget = min(8.0, DEADLINE - 15 - (time.time() - start_time))
        if cg_budget > 2.0:
            c = _contact_graph_refine(best, time_budget=cg_budget)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 4) Smooth soft-min polish from the best layout.
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=0.8, max_iter=1000))
    
    for k in (80.0, 150.0, 250.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            c = _softmin_refine(c0, k, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 5) Final SLSQP epigraph polish with tight tolerance
    if time.time() - start_time < DEADLINE - 5:
        final_budget = min(15.0, DEADLINE - 5 - (time.time() - start_time))
        if final_budget > 3.0:
            c = _slsqp_epigraph(best, best_r, time_budget=final_budget)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 6) Final contact-graph micro-refinement
    if time.time() - start_time < DEADLINE - 2:
        c = _contact_graph_refine(best, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
