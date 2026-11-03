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
                       options={'maxiter': 120, 'maxfun': 3000,
                                'ftol': 1e-13, 'gtol': 1e-9,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _langevin_optimize(c0, r_target, time_budget=6.0):
    """
    Langevin dynamics with cooling to find a dense packing.
    Uses a potential that penalizes overlaps and encourages filling space.
    """
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    
    start = time.time()
    dt = 0.002
    gamma = 0.1  # friction
    k_b = 1.0  # "temperature"
    
    # Start with a moderate temperature and cool down
    T = 0.01
    T_min = 0.0001
    cool_rate = 0.999
    
    # Adaptive step size
    max_steps = 100000
    
    for step in range(max_steps):
        if time.time() - start > time_budget:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (soft potential)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        # Only consider close pairs for efficiency
        close = d < 3 * r_target
        if np.any(close):
            safe_d = np.where(d > 1e-10, d, 1e-10)
            # Soft repulsion: F = -dU/dr, U = 0.5 * k * (r_target - d)^2 if d < r_target
            # But we want to push apart even if slightly overlapping
            overlap = r_target - d
            mask = overlap > 0
            if np.any(mask):
                # Force magnitude proportional to overlap
                f_mag = 10.0 * overlap
                vec = (c[:, None, :] - c[None, :, :]) / safe_d[:, :, None]
                f_vec = np.where(mask[:, :, None], f_mag[:, :, None] * vec, 0.0)
                forces += f_vec.sum(axis=1)
        
        # Wall forces
        margin = 0.01
        for axis in [0, 1]:
            # Right/Top wall
            dist_r = 0.5 - c[:, axis]
            pen_r = np.clip(margin - dist_r, 0, None)
            forces[:, axis] += 10.0 * pen_r
            
            # Left/Bottom wall
            dist_l = c[:, axis] + 0.5
            pen_l = np.clip(margin - dist_l, 0, None)
            forces[:, axis] -= 10.0 * pen_l
        
        # Langevin update: m*a = F + gamma*noise - gamma*v (simplified)
        # v = (v + dt*(F + noise))/ (1 + dt*gamma)  -- implicit Euler
        noise = np.random.normal(0, np.sqrt(2 * T * dt), c.shape)
        c = c + dt * (forces + noise) / (1 + dt * gamma)
        
        # Clamp to valid region
        c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
        
        # Cool down
        T *= cool_rate
        if T < T_min:
            T = T_min
            
        # Occasionally check if we can increase radius
        if step % 5000 == 0:
            pass # Could try to grow r here, but keep it simple for now
    
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(8):
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
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 15:
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

    # 2) Langevin optimization from best and a few others
    for c0 in [best] + inits[:3]:
        if time.time() - start_time > DEADLINE - 10:
            break
        # Try to optimize at current best radius
        cur = _langevin_optimize(c0, best_r, time_budget=3.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 3) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.5, max_iter=2000))
    for c0 in inits[:3]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.5, max_iter=2000))

    for k in (80.0, 150.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 6:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 4) Final Langevin polish at the best radius found
    if time.time() - start_time < DEADLINE - 5:
        cur = _langevin_optimize(best, best_r, time_budget=4.0)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 5) Final soft-min polish
    if time.time() - start_time < DEADLINE - 3:
        c = _softmin_refine(best, 200.0, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
