"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2.

Strategy: contact-graph based optimization.
1. Generate candidate contact graphs (which circles touch which, and which touch walls).
2. For each contact graph, solve the under/over-determined system of tangency equations
   to find positions and radii.
3. Use a penalty-based nonlinear solver to maximize sum of radii subject to the
   contact constraints being approximately satisfied.
4. Combine with a final local optimization pass (L-BFGS-B) to polish.
"""
import time
import numpy as np
from scipy.optimize import minimize, fsolve
from itertools import combinations

N = 26
_DEADLINE = 150.0
_START = time.time()
_EPS = 1e-10


def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0


def _feasible_score(pos, rad):
    """Return (value, feasible). value = sum(rad) if feasible else -inf."""
    n = len(rad)
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d2 = dx * dx + dy * dy
            rr = rad[i] + rad[j]
            if d2 < rr * rr - _EPS:
                return -np.inf, False
    return float(np.sum(rad)), True


def _project_feasible(pos, rad, iters=30):
    """Heuristic repair: shrink radii / nudge to satisfy constraints."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        changed = False
        for _ in range(5):
            inner_changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - _EPS:
                        s = (d - _EPS) / (need + _EPS) if need > 0 else 1.0
                        rad[i] = max(rad[i] * s, 0.0)
                        rad[j] = max(rad[j] * s, 0.0)
                        inner_changed = True
            if not inner_changed:
                break
            changed = True
        if not changed:
            break
    val, feas = _feasible_score(pos, rad)
    return pos, rad, feas


def _soft_objective(z, lam):
    """z = [x0,y0,r0, x1,y1,r1, ...]. Minimize -sum(r) + lam*(overlap + wall penalties)."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        pen += np.sum((needu[mask] - du) ** 2)
    return base + lam * pen


def _soft_grad(z, lam):
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    gr -= 1.0
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        fu = needu[mask] - du
        inv_u = 1.0 / du
        np.add.at(gx, idx_i, lam * 2.0 * fu * (-dxu * inv_u))
        np.add.at(gx, idx_j, lam * 2.0 * fu * (dxu * inv_u))
        np.add.at(gy, idx_i, lam * 2.0 * fu * (-dyu * inv_u))
        np.add.at(gy, idx_j, lam * 2.0 * fu * (dyu * inv_u))
        np.add.at(gr, idx_i, lam * 2.0 * fu)
        np.add.at(gr, idx_j, lam * 2.0 * fu)
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _hex_seed(jitter=0.0, rng=None):
    """Generate a hexagonal-ish arrangement scaled to fit."""
    rng = rng or np.random.default_rng()
    pts = []
    rows = 0
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append((x, y))
            col += 1
        rows += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    rad = np.full(N, 0.2)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.98
    return pts, rad


def _contact_graph_objective(z, contacts, wall_contacts, lam_contact=100.0, lam_wall=100.0):
    """
    Objective for a given contact graph.
    z = [x0,y0,r0, ..., x25,y25,r25]
    contacts: list of (i, j) pairs that should be tangent
    wall_contacts: dict mapping circle index to set of walls {'L','R','B','T'}
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    obj = -np.sum(r)
    
    # Contact constraints: distance between i and j should equal r_i + r_j
    for (i, j) in contacts:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        obj += lam_contact * (d - (r[i] + r[j])) ** 2
    
    # Wall contact constraints
    for i, walls in wall_contacts.items():
        if 'L' in walls:
            obj += lam_wall * (x[i] - r[i]) ** 2
        if 'R' in walls:
            obj += lam_wall * (1.0 - x[i] - r[i]) ** 2
        if 'B' in walls:
            obj += lam_wall * (y[i] - r[i]) ** 2
        if 'T' in walls:
            obj += lam_wall * (1.0 - y[i] - r[i]) ** 2
    
    # Non-contact separation (soft, to prevent overlaps of non-contact pairs)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in contacts or (j, i) in contacts:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            need = r[i] + r[j]
            if d < need:
                pen += (need - d) ** 2
    obj += 10.0 * pen
    
    # Wall non-contact (soft, to keep circles inside)
    pen_wall = 0.0
    for i in range(n):
        pen_wall += np.maximum(r[i] - x[i], 0.0) ** 2
        pen_wall += np.maximum(x[i] + r[i] - 1.0, 0.0) ** 2
        pen_wall += np.maximum(r[i] - y[i], 0.0) ** 2
        pen_wall += np.maximum(y[i] + r[i] - 1.0, 0.0) ** 2
    obj += 50.0 * pen_wall
    
    return obj


def _contact_graph_grad(z, contacts, wall_contacts, lam_contact=100.0, lam_wall=100.0):
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    gr -= 1.0
    
    # Contact gradients
    for (i, j) in contacts:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        if d < 1e-12:
            continue
        err = d - (r[i] + r[j])
        gx[i] += 2 * lam_contact * err * (dx / d)
        gx[j] += 2 * lam_contact * err * (-dx / d)
        gy[i] += 2 * lam_contact * err * (dy / d)
        gy[j] += 2 * lam_contact * err * (-dy / d)
        gr[i] += 2 * lam_contact * err * (-1.0)
        gr[j] += 2 * lam_contact * err * (-1.0)
    
    # Wall contact gradients
    for i, walls in wall_contacts.items():
        if 'L' in walls:
            err = x[i] - r[i]
            gx[i] += 2 * lam_wall * err
            gr[i] += 2 * lam_wall * err * (-1.0)
        if 'R' in walls:
            err = 1.0 - x[i] - r[i]
            gx[i] += 2 * lam_wall * err * (-1.0)
            gr[i] += 2 * lam_wall * err * (-1.0)
        if 'B' in walls:
            err = y[i] - r[i]
            gy[i] += 2 * lam_wall * err
            gr[i] += 2 * lam_wall * err * (-1.0)
        if 'T' in walls:
            err = 1.0 - y[i] - r[i]
            gy[i] += 2 * lam_wall * err * (-1.0)
            gr[i] += 2 * lam_wall * err * (-1.0)
    
    # Non-contact separation gradient
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in contacts or (j, i) in contacts:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            need = r[i] + r[j]
            if d < need and d > 1e-12:
                err = need - d
                gx[i] += 20.0 * err * (dx / d)
                gx[j] += 20.0 * err * (-dx / d)
                gy[i] += 20.0 * err * (dy / d)
                gy[j] += 20.0 * err * (-dy / d)
                gr[i] += 20.0 * err
                gr[j] += 20.0 * err
    
    # Wall non-contact gradient
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += 100.0 * m[mpos] * (-1.0)
        gr[mpos] += 100.0 * m[mpos]
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += 100.0 * m[mpos]
        gr[mpos] += 100.0 * m[mpos]
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += 100.0 * m[mpos] * (-1.0)
        gr[mpos] += 100.0 * m[mpos]
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += 100.0 * m[mpos]
        gr[mpos] += 100.0 * m[mpos]
    
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _build_contact_graph_from_packing(pos, rad, tol=0.05):
    """
    Given a packing, determine which circles are in contact (distance ≈ r_i + r_j)
    and which touch walls.
    """
    n = N
    contacts = set()
    wall_contacts = {}
    
    for i in range(n):
        walls = set()
        if abs(pos[i, 0] - rad[i]) < tol:
            walls.add('L')
        if abs(1.0 - pos[i, 0] - rad[i]) < tol:
            walls.add('R')
        if abs(pos[i, 1] - rad[i]) < tol:
            walls.add('B')
        if abs(1.0 - pos[i, 1] - rad[i]) < tol:
            walls.add('T')
        if walls:
            wall_contacts[i] = walls
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            need = rad[i] + rad[j]
            if abs(d - need) < tol:
                contacts.add((i, j))
    
    return list(contacts), wall_contacts


def _solve_contact_graph(pos0, rad0, contacts, wall_contacts, max_iter=500):
    """
    Solve for optimal positions and radii given a contact graph.
    """
    n = N
    z0 = np.empty(3 * n)
    z0[0::3] = pos0[:, 0]
    z0[1::3] = pos0[:, 1]
    z0[2::3] = rad0
    
    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N
    
    res = minimize(
        _contact_graph_objective, z0,
        jac=lambda z: _contact_graph_grad(z, contacts, wall_contacts),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    
    z = res.x
    p = np.column_stack([z[0::3], z[1::3]])
    r = z[2::3]
    return p, r


def _local_polish(pos, rad, max_iter=300):
    """Final local optimization without contact graph constraints."""
    n = N
    z0 = np.empty(3 * n)
    z0[0::3] = pos[:, 0]
    z0[1::3] = pos[:, 1]
    z0[2::3] = rad
    
    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N
    
    res = minimize(
        _soft_objective, z0,
        jac=_soft_grad,
        method='L-BFGS-B',
        bounds=bounds,
        args=(0.1,),
        options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12}
    )
    
    z = res.x
    p = np.column_stack([z[0::3], z[1::3]])
    r = z[2::3]
    return p, r


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(42)
    
    best_val = -np.inf
    best_pos = None
    best_rad = None
    
    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N
    
    def _update_best(p, r):
        nonlocal best_val, best_pos, best_rad
        p, r, feas = _project_feasible(p, r)
        if feas:
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_pos = p.copy()
                best_rad = r.copy()
    
    # Phase 1: Generate diverse initial packings and run contact-graph optimization
    # Start with hexagonal seeds
    for jitter in [0.0, 0.02, 0.05, 0.1, 0.15]:
        if _remaining() < 30.0:
            break
        p0, r0 = _hex_seed(jitter=jitter, rng=rng)
        # Project to feasibility first
        p0, r0, feas = _project_feasible(p0, r0)
        if not feas:
            continue
        
        # Extract contact graph
        contacts, wall_contacts = _build_contact_graph_from_packing(p0, r0, tol=0.08)
        
        # Solve with contact graph
        p_sol, r_sol = _solve_contact_graph(p0, r0, contacts, wall_contacts, max_iter=400)
        _update_best(p_sol, r_sol)
        
        # Local polish
        if _remaining() < 15.0:
            continue
        p_pol, r_pol = _local_polish(p_sol, r_sol, max_iter=200)
        _update_best(p_pol, r_pol)
    
    # Phase 2: Random restarts with contact graph
    if best_pos is not None:
        for trial in range(15):
            if _remaining() < 20.0:
                break
            # Perturb best solution
            p0 = best_pos + rng.normal(0, 0.04, size=best_pos.shape)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = best_rad * rng.uniform(0.85, 1.0, size=N)
            
            p0, r0, feas = _project_feasible(p0, r0)
            if not feas:
                continue
            
            contacts, wall_contacts = _build_contact_graph_from_packing(p0, r0, tol=0.06)
            
            p_sol, r_sol = _solve_contact_graph(p0, r0, contacts, wall_contacts, max_iter=300)
            _update_best(p_sol, r_sol)
            
            if _remaining() < 10.0:
                continue
            p_pol, r_pol = _local_polish(p_sol, r_sol, max_iter=150)
            _update_best(p_pol, r_pol)
    
    # Phase 3: Pure L-BFGS-B with annealing from best
    if best_pos is not None and _remaining() > 20.0:
        for lam in [5.0, 2.0, 0.5, 0.1, 0.05]:
            if _remaining() < 10.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            _update_best(p, r)
    
    # Phase 4: Final high-precision polish
    if best_pos is not None and _remaining() > 10.0:
        for lam in [0.02, 0.01, 0.005]:
            if _remaining() < 5.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            _update_best(p, r)
    
    # Fallback
    if best_pos is None:
        p, r = _hex_seed(jitter=0.0)
        p, r, feas = _project_feasible(p, r)
        if feas:
            best_pos, best_rad = p, r
        else:
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            arr = np.array(cells[:N])
            return arr
        best_pos, best_rad = p, r
    
    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
