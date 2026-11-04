"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
_START = time.time()
_EPS = 1e-9


def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin


def _feasible_score(pos, rad):
    """Return (value, feasible). value = sum(rad) if feasible else -inf."""
    n = len(rad)
    # wall check
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    # pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d2 = dx * dx + dy * dy
            rr = rad[i] + rad[j]
            if d2 < rr * rr - _EPS:
                return -np.inf, False
    return float(np.sum(rad)), True


def _maximize_radii(pos, rad):
    """
    For fixed positions, maximize each radius such that:
    1. Circle i fits in the box.
    2. Circle i does not overlap with any other circle j.
    Uses iterative relaxation.
    """
    n = len(rad)
    pos = pos.copy()
    rad = rad.copy()
    
    # Precompute pairwise distances
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    # Wall constraints for each circle
    wall_max = np.minimum(
        np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
        np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    )
    
    rad = np.maximum(rad, 1e-10)
    
    # Iterative relaxation
    for iteration in range(200):
        changed = False
        for i in range(n):
            max_r = wall_max[i]
            for j in range(n):
                if i != j:
                    limit = dist[i, j] - rad[j]
                    if limit < max_r:
                        max_r = limit
            
            max_r = max(0.0, max_r)
            
            if abs(max_r - rad[i]) > 1e-10:
                rad[i] = 0.5 * rad[i] + 0.5 * max_r
                changed = True
            else:
                rad[i] = max_r
        
        if not changed:
            break
    
    # Final exact pass
    for i in range(n):
        max_r = wall_max[i]
        for j in range(n):
            if i != j:
                limit = dist[i, j] - rad[j]
                if limit < max_r:
                    max_r = limit
        rad[i] = max(0.0, max_r)
    
    return pos, rad


def _physics_relax(pos, rad, steps=200, dt=0.01, damping=0.9):
    """
    Use force-directed dynamics to pack circles tightly.
    Forces:
    - Repulsion if overlapping.
    - Weak attraction if separated (to keep them together/packed).
    - Wall constraints (hard boundaries with soft penalty).
    """
    pos = pos.copy()
    rad = rad.copy()
    vel = np.zeros_like(pos)
    n = len(pos)
    
    for _ in range(steps):
        forces = np.zeros_like(pos)
        
        # Pairwise forces
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        # Target distance is r_i + r_j
        # If dist < target, repel. If dist > target, attract (weakly).
        target = rad[:, None] + rad[None, :]
        
        # Force magnitude
        # F = k * (dist - target)
        # We want to minimize (dist - target)^2
        # Gradient w.r.t pos_i is 2*(dist-target)*unit_vec
        # Let's use a simple spring force
        k_rep = 10.0
        k_att = 0.1
        
        # Vectorized force calculation
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-6, 1e-6, dist)
        unit_x = dx / dist_safe
        unit_y = dy / dist_safe
        
        diff = dist - target
        # Repulsive force if diff < 0, attractive if diff > 0
        # We only want repulsion to be strong, attraction weak to prevent clumping too much?
        # Actually, for packing, we want them to touch. So attraction is good.
        
        force_mag = np.zeros_like(dist)
        mask_rep = diff < 0
        mask_att = diff > 0
        
        force_mag[mask_rep] = k_rep * diff[mask_rep]
        force_mag[mask_att] = k_att * diff[mask_att]
        
        # Force on i due to j is in direction of j from i?
        # If dist < target (overlap), force should push i away from j.
        # Vector from i to j is (pos_j - pos_i).
        # Our dx is pos_i - pos_j. So unit_x points from j to i.
        # If overlapping (diff < 0), we want force in direction of unit_x (away from j).
        # So force_x += force_mag * unit_x.
        
        fx = np.sum(force_mag * unit_x, axis=1)
        fy = np.sum(force_mag * unit_y, axis=1)
        
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        # Wall forces
        # If x < r, push right. If x > 1-r, push left.
        # Soft wall
        k_wall = 5.0
        v_left = np.maximum(rad - pos[:, 0], 0.0)
        v_right = np.maximum(pos[:, 0] + rad - 1.0, 0.0)
        v_bot = np.maximum(rad - pos[:, 1], 0.0)
        v_top = np.maximum(pos[:, 1] + rad - 1.0, 0.0)
        
        forces[:, 0] += k_wall * (v_left - v_right)
        forces[:, 1] += k_wall * (v_bot - v_top)
        
        # Update velocity and position
        vel = vel * damping + forces * dt
        pos += vel * dt
        
        # Hard clip to box to prevent explosion
        # Note: clipping positions breaks momentum, but keeps things stable
        # We must ensure they don't stick to walls in a bad way.
        # For now, just clip.
        
    return pos, rad


def _soft_objective(z, lam):
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    
    v_left = np.maximum(r - x, 0.0)
    v_right = np.maximum(x + r - 1.0, 0.0)
    v_bot = np.maximum(r - y, 0.0)
    v_top = np.maximum(y + r - 1.0, 0.0)
    
    pen += np.sum(v_left ** 2) + np.sum(v_right ** 2) + np.sum(v_bot ** 2) + np.sum(v_top ** 2)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    
    du = np.sqrt(np.maximum(d2u, 1e-12))
    v_ov = np.maximum(needu - du, 0.0)
    pen += np.sum(v_ov ** 2)
    
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
    
    v_left = np.maximum(r - x, 0.0)
    if np.any(v_left > 0):
        gx[v_left > 0] += lam * 2.0 * v_left[v_left > 0] * (-1.0)
        gr[v_left > 0] += lam * 2.0 * v_left[v_left > 0]
        
    v_right = np.maximum(x + r - 1.0, 0.0)
    if np.any(v_right > 0):
        gx[v_right > 0] += lam * 2.0 * v_right[v_right > 0] * (1.0)
        gr[v_right > 0] += lam * 2.0 * v_right[v_right > 0]
        
    v_bot = np.maximum(r - y, 0.0)
    if np.any(v_bot > 0):
        gy[v_bot > 0] += lam * 2.0 * v_bot[v_bot > 0] * (-1.0)
        gr[v_bot > 0] += lam * 2.0 * v_bot[v_bot > 0]
        
    v_top = np.maximum(y + r - 1.0, 0.0)
    if np.any(v_top > 0):
        gy[v_top > 0] += lam * 2.0 * v_top[v_top > 0] * (1.0)
        gr[v_top > 0] += lam * 2.0 * v_top[v_top > 0]
        
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    
    du = np.sqrt(np.maximum(d2u, 1e-12))
    v_ov = np.maximum(needu - du, 0.0)
    
    mask = v_ov > 0
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du_m = du[mask]
        v_m = v_ov[mask]
        
        inv_u = 1.0 / du_m
        np.add.at(gx, idx_i, lam * 2.0 * v_m * (-dxu * inv_u))
        np.add.at(gx, idx_j, lam * 2.0 * v_m * (dxu * inv_u))
        np.add.at(gy, idx_i, lam * 2.0 * v_m * (-dyu * inv_u))
        np.add.at(gy, idx_j, lam * 2.0 * v_m * (dyu * inv_u))
        np.add.at(gr, idx_i, lam * 2.0 * v_m)
        np.add.at(gr, idx_j, lam * 2.0 * v_m)
        
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    def _update_best(p, r):
        nonlocal best_val, best_pos, best_rad
        val, feas = _feasible_score(p, r)
        if feas:
            if val > best_val:
                best_val = val
                best_pos = p.copy()
                best_rad = r.copy()
                return True
        return False

    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N

    # Generate seeds
    seeds = []
    # Hex seed
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
    rad_init = np.full(N, 0.1)
    seeds.append((pts, rad_init))
    
    # Random seeds
    for _ in range(5):
        p = rng.uniform(0.2, 0.8, size=(N, 2))
        r = np.full(N, 0.1)
        seeds.append((p, r))

    # Optimization Loop
    for si, (pos, rad) in enumerate(seeds):
        if _remaining() < 15.0:
            break
            
        cur_pos, cur_rad = pos.copy(), rad.copy()
        
        # 1. Physics Relaxation to pack tightly
        cur_pos, cur_rad = _physics_relax(cur_pos, cur_rad, steps=100)
        
        # 2. Maximize Radii
        cur_pos, cur_rad = _maximize_radii(cur_pos, cur_rad)
        _update_best(cur_pos, cur_rad)
        
        # 3. L-BFGS Polish
        if _remaining() < 5.0:
            continue
            
        z = np.empty(3 * N)
        z[0::3] = cur_pos[:, 0]
        z[1::3] = cur_pos[:, 1]
        z[2::3] = cur_rad
        
        # Use a moderate lambda to allow some growth
        res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                       bounds=bounds, args=(10.0,),
                       options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
        
        z = res.x
        p = np.column_stack([z[0::3], z[1::3]])
        r = z[2::3]
        
        # 4. Final Maximize Radii
        p, r = _maximize_radii(p, r)
        _update_best(p, r)

    # Final Polish on Best
    if best_pos is not None and _remaining() > 10.0:
        z = np.empty(3 * N)
        z[0::3] = best_pos[:, 0]
        z[1::3] = best_pos[:, 1]
        z[2::3] = best_rad
        
        res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                       bounds=bounds, args=(1.0,),
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        
        z = res.x
        p = np.column_stack([z[0::3], z[1::3]])
        r = z[2::3]
        p, r = _maximize_radii(p, r)
        _update_best(p, r)

    if best_pos is None:
        # Fallback
        p, r = _physics_relax(rng.uniform(0.2, 0.8, size=(N, 2)), np.full(N, 0.1))
        p, r = _maximize_radii(p, r)
        _update_best(p, r)
        
    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
