"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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


def _delaunay_contact_graph(c, r, threshold=2.1):
    """Return set of (i,j) pairs that are near-contact in the Delaunay graph."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    edges.add((min(i, j), max(i, j)))
        # Filter to only near-contact edges (within threshold*r)
        near = set()
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < threshold * r:
                near.add((i, j))
        return near
    except Exception:
        # Fallback: all pairs within threshold*r
        near = set()
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < threshold * r:
                    near.add((i, j))
        return near


def _lattice_init():
    """Initialize 50 points on a hexagonal lattice with basis offsets.
    
    Uses a 7x7 grid (49 pts) + 1 extra, parameterized as:
    center_i = origin + i*a1 + j*a2 + offset_ij
    
    Parameters: origin(2), a1(2), a2(2), offsets(50*2) = 106 vars
    But we'll use a simpler approach: 7x7 grid with per-point offsets.
    """
    m = 7
    # Base hexagonal lattice
    a1 = np.array([1.0 / m, 0.0])
    a2 = np.array([0.5 / m, np.sqrt(3) / 2 / m])
    origin = np.array([-0.5, -0.5])
    
    pts = []
    for j in range(m):
        for i in range(m):
            pts.append(origin + i * a1 + j * a2)
    # 50th point: place at center or a gap
    pts.append(np.array([0.0, 0.0]))
    return np.array(pts)


def _lattice_objective(params, m=7):
    """Compute radius from lattice parameters.
    
    params layout:
      [0:2]   = origin (ox, oy)
      [2:4]   = a1 vector
      [4:6]   = a2 vector
      [6:6+2*N] = offsets for each of N points
      
    Total: 6 + 2*N = 106 parameters
    """
    ox, oy = params[0], params[1]
    a1x, a1y = params[2], params[3]
    a2x, a2y = params[4], params[5]
    offsets = params[6:6 + 2 * N].reshape(N, 2)
    
    # Generate base lattice positions (7x7 = 49 + 1 special)
    base = np.zeros((N, 2))
    idx = 0
    for j in range(m):
        for i in range(m):
            base[idx] = np.array([ox + i * a1x + j * a2x,
                                  oy + i * a1y + j * a2y])
            idx += 1
    # 50th point: use a different rule - center of the square
    base[49] = np.array([0.0, 0.0])
    
    centers = base + offsets
    
    # Compute packing radius
    r = np.inf
    # Wall distances
    w = np.min(np.stack([
        0.5 - centers[:, 0], centers[:, 0] + 0.5,
        0.5 - centers[:, 1], centers[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    
    # Pairwise distances
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    
    return r


def _lattice_optimize(c_init, time_budget=8.0, maxiter=500, maxfun=20000):
    """Optimize lattice parameters to maximize packing radius.
    
    Starts from a good initial layout and fits lattice parameters.
    """
    c_init = np.clip(np.asarray(c_init, dtype=float), -0.5, 0.5)
    m = 7
    
    # Initialize lattice parameters from the current layout
    # Estimate origin, a1, a2 from first few points
    if N >= 9:
        ox, oy = c_init[0, 0], c_init[0, 1]
        a1 = c_init[1] - c_init[0]
        a2 = c_init[m] - c_init[0]
    else:
        ox, oy = 0.0, 0.0
        a1 = np.array([1.0 / m, 0.0])
        a2 = np.array([0.5 / m, np.sqrt(3) / 2 / m])
    
    # Compute initial offsets
    base = np.zeros((N, 2))
    idx = 0
    for j in range(m):
        for i in range(m):
            base[idx] = np.array([ox + i * a1[0] + j * a2[0],
                                  oy + i * a1[1] + j * a2[1]])
            idx += 1
    base[49] = np.array([0.0, 0.0])
    offsets = c_init - base
    
    x0 = np.concatenate([[ox, oy], a1, a2, offsets.ravel()])
    
    def neg_radius(x):
        return -_lattice_objective(x, m=m)
    
    def neg_radius_grad(x):
        """Numerical gradient (sparse, only for changed params)."""
        eps = 1e-7
        g = np.zeros_like(x)
        # Only compute gradient for lattice params (first 6) and a subset of offsets
        # For efficiency, use finite differences on the full vector
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (neg_radius(xp) - neg_radius(xm)) / (2 * eps)
        return g
    
    try:
        start = time.time()
        res = minimize(neg_radius, x0, method='L-BFGS-B',
                       jac=neg_radius_grad,
                       bounds=[(-0.5, 0.5)] * 6 + [(-0.3, 0.3)] * (2 * N),
                       options={'maxiter': maxiter, 'maxfun': maxfun,
                                'ftol': 1e-15, 'gtol': 1e-10, 'iprint': 0})
        if time.time() - start > time_budget:
            return c_init
        
        # Extract centers from optimized parameters
        ox, oy = res.x[0], res.x[1]
        a1x, a1y = res.x[2], res.x[3]
        a2x, a2y = res.x[4], res.x[5]
        offsets = res.x[6:6 + 2 * N].reshape(N, 2)
        
        base = np.zeros((N, 2))
        idx = 0
        for j in range(m):
            for i in range(m):
                base[idx] = np.array([ox + i * a1x + j * a2x,
                                      oy + i * a1y + j * a2y])
                idx += 1
        base[49] = np.array([0.0, 0.0])
        centers = base + offsets
        centers = np.clip(centers, -0.5, 0.5)
        return centers
    except Exception:
        return c_init


def _slsqp_epigraph(c0, r0, time_budget=5.0, maxiter=200, ftol=1e-14):
    """SLSQP epigraph formulation: maximize t subject to
    t <= |c_i - c_j|/2 for all near-contact pairs,
    t <= wall distances for all points.
    Uses sparse constraints from Delaunay contact graph."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)
    
    # Get contact graph with tighter threshold for better convergence
    contacts = _delaunay_contact_graph(c0, r0, threshold=2.1)
    n_contacts = len(contacts)
    
    def obj(x):
        # x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, t]
        return -x[-1]  # minimize -t = maximize t
    
    def obj_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g
    
    # Inequality constraints: t <= d_ij/2 and t <= wall_i
    # We'll use SLSQP with constraints in form c(x) >= 0
    
    def constraints(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[-1]
        cons = []
        # Pairwise constraints for near-contact pairs
        for i, j in contacts:
            d = np.linalg.norm(pts[i] - pts[j])
            cons.append(d / 2.0 - t)
        # Wall constraints (use min/max for efficiency)
        cons.append(pts[:, 0].min() + 0.5 - t)
        cons.append(0.5 - pts[:, 0].max() - t)
        cons.append(pts[:, 1].min() + 0.5 - t)
        cons.append(0.5 - pts[:, 1].max() - t)
        return cons
    
    def constraints_jac(x):
        pts = x[:2*N].reshape(N, 2)
        n_cons = n_contacts + 4
        J = np.zeros((n_cons, 2 * N + 1))
        
        # Pairwise constraint gradients
        for idx, (i, j) in enumerate(contacts):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                continue
            u = (pts[i] - pts[j]) / d
            J[idx, 2*i] = 0.5 * u[0]
            J[idx, 2*i+1] = 0.5 * u[1]
            J[idx, 2*j] = -0.5 * u[0]
            J[idx, 2*j+1] = -0.5 * u[1]
            J[idx, -1] = -1.0
        
        # Wall constraint gradients
        base = n_contacts
        min_idx_x = np.argmin(pts[:, 0])
        J[base, 2*min_idx_x] = 1.0
        J[base, -1] = -1.0
        
        max_idx_x = np.argmax(pts[:, 0])
        J[base+1, 2*max_idx_x] = -1.0
        J[base+1, -1] = -1.0
        
        min_idx_y = np.argmin(pts[:, 1])
        J[base+2, 2*min_idx_y+1] = 1.0
        J[base+2, -1] = -1.0
        
        max_idx_y = np.argmax(pts[:, 1])
        J[base+3, 2*max_idx_y+1] = -1.0
        J[base+3, -1] = -1.0
        
        return J
    
    x0 = np.concatenate([c0.ravel(), [r0 * 0.99]])  # Start slightly below to ensure feasibility
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    cons_list = []
    for i in range(n_contacts + 4):
        cons_list.append({'type': 'ineq', 'fun': lambda x, idx=i: constraints(x)[idx],
                         'jac': lambda x, idx=i: constraints_jac(x)[idx]})
    
    try:
        start = time.time()
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons_list,
                       options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0})
        if time.time() - start > time_budget:
            return c0
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
        return c
    except Exception:
        return c0


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init(), _lattice_init()]
    rng = np.random.default_rng(12345)
    for _ in range(4):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 15:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(6):
            if time.time() - start_time > DEADLINE - 15:
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

    # 2) Lattice-based optimization (new: reduced dimensionality)
    if time.time() - start_time < DEADLINE - 20:
        c = _lattice_optimize(best, time_budget=8.0, maxiter=400, maxfun=15000)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
        else:
            # Try lattice from hex init too
            c = _lattice_optimize(_hex_init(), time_budget=5.0, maxiter=300, maxfun=10000)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 3) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))
    for c0 in inits[:3]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=1500))

    # Adaptive k schedule: start loose, tighten progressively
    k_schedule = [80.0, 150.0, 250.0, 400.0]
    for k in k_schedule:
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 12:
                break
            c = _softmin_refine(c0, k, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                candidates[0] = best

    # 4) Second lattice pass after soft-min (refit lattice to improved layout)
    if time.time() - start_time < DEADLINE - 15:
        c = _lattice_optimize(best, time_budget=6.0, maxiter=350, maxfun=12000)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Delaunay-based SLSQP epigraph polish on the best layout
    if time.time() - start_time < DEADLINE - 10:
        c = _slsqp_epigraph(best, best_r, time_budget=5.0, maxiter=150, ftol=1e-14)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Final high-precision polish on the best layout with tighter tolerance
    if time.time() - start_time < DEADLINE - 7:
        c = _softmin_refine(best, 350.0, time_budget=2.5,
                            maxiter=350, maxfun=10000,
                            ftol=1e-17, gtol=1e-11)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 7) Third lattice refinement
    if time.time() - start_time < DEADLINE - 8:
        c = _lattice_optimize(best, time_budget=4.0, maxiter=250, maxfun=8000)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 8) Second SLSQP epigraph pass after soft-min polish
    if time.time() - start_time < DEADLINE - 4:
        c = _slsqp_epigraph(best, best_r, time_budget=3.0, maxiter=120, ftol=1e-15)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 9) Ultra-high-precision final pass
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 500.0, time_budget=1.5,
                            maxiter=250, maxfun=8000,
                            ftol=1e-18, gtol=1e-13)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
