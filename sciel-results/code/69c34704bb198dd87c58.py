"""Rigid-body dynamics packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
TIME_LIMIT = 118.0  # safety margin before 150s kill


def _hex_init(n=N, spacing=0.098):
    """Dense hexagonal lattice clipped to unit disk, keep n closest to center."""
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _hex_init(n, spacing * 0.9)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _md_simulate(pts, start_time, time_limit):
    """
    Molecular dynamics with purely repulsive truncated power-law potential.
    
    Force model:
      - Pair (i,j): if d < cutoff, F = strength * (cutoff - d)^power along separation
      - Wall (i):   if |c_i| > wall_cutoff, F = wall_strength * (|c_i| - wall_cutoff)^wall_power inward
    
    Integration: Velocity Verlet with adaptive dt and velocity damping.
    """
    n = len(pts)
    pos = pts.copy()
    vel = np.zeros_like(pos)
    force = np.zeros_like(pos)
    
    # Parameters
    cutoff = 0.115          # slightly above expected 2r (~0.1063)
    strength = 50.0
    power = 2.0             # quadratic repulsion (softer than LJ 12-6)
    wall_cutoff = 0.985     # start pushing before boundary
    wall_strength = 80.0
    wall_power = 2.0
    
    dt = 0.0015
    damping = 0.995         # velocity damping per step
    max_dt = 0.003
    min_dt = 0.0002
    
    best_pos = pos.copy()
    best_score = _score(pos)
    no_improve = 0
    max_no_improve = 2000
    it = 0
    
    # Precompute for efficiency
    idx_i, idx_j = np.triu_indices(n, k=1)
    pairs_count = len(idx_i)
    
    def compute_forces(p):
        """Compute repulsive forces for all pairs and walls."""
        f = np.zeros_like(p)
        
        # Pair forces
        diff = p[idx_i] - p[idx_j]  # (pairs, 2)
        d = np.hypot(diff[:, 0], diff[:, 1])  # (pairs,)
        mask = d < cutoff
        if np.any(mask):
            dm = d[mask]
            diffm = diff[mask]
            # Force magnitude: strength * (cutoff - d)^power
            fm = strength * (cutoff - dm) ** power
            # Direction: push apart -> force on i is +diff/d, on j is -diff/d
            unit = diffm / dm[:, None]
            f[idx_i[mask]] += (fm * unit[:, 0])[:, None] * np.ones((1, 2))
            f[idx_i[mask]] += (fm * unit[:, 1])[:, None] * np.ones((1, 2)) * np.array([[0, 0], [1, 0]])
            # Actually let's do this more carefully:
            # f_i += fm * unit (push i away from j)
            # f_j -= fm * unit (push j away from i)
            # Let me redo:
        
        # Redo pair forces cleanly
        f = np.zeros_like(p)
        diff = p[idx_i] - p[idx_j]
        d = np.hypot(diff[:, 0], diff[:, 1])
        mask = d < cutoff
        if np.any(mask):
            dm = d[mask]
            diffm = diff[mask]
            fm = strength * (cutoff - dm) ** power  # scalar per pair
            unit = diffm / dm[:, None]  # (m, 2)
            # Force on i: +fm * unit (away from j)
            f[idx_i[mask]] += fm[:, None] * unit
            # Force on j: -fm * unit (away from i)
            f[idx_j[mask]] -= fm[:, None] * unit
        
        # Wall forces
        rad = np.hypot(p[:, 0], p[:, 1])
        wall_mask = rad > wall_cutoff
        if np.any(wall_mask):
            rm = rad[wall_mask]
            pm = p[wall_mask]
            fw = wall_strength * (rm - wall_cutoff) ** wall_power
            inward = -pm / rm[:, None]  # direction toward center
            f[wall_mask] += fw[:, None] * inward
        
        return f
    
    # Initial forces
    force = compute_forces(pos)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break
        
        # Velocity Verlet
        # pos_new = pos + vel*dt + 0.5*force/m*dt^2  (m=1)
        pos += vel * dt + 0.5 * force * dt * dt
        vel *= damping  # apply damping
        
        # Compute new forces
        force_new = compute_forces(pos)
        
        # Update velocity
        vel += 0.5 * (force + force_new) * dt
        vel *= damping
        force = force_new
        
        # Adaptive dt: reduce if max force is large
        max_f = np.max(np.linalg.norm(force, axis=1))
        if max_f > 5.0:
            dt = max(min_dt, dt * 0.7)
        elif max_f < 0.5 and dt < max_dt:
            dt = min(max_dt, dt * 1.1)
        
        # Safety: clamp inside unit circle
        rad = np.hypot(pos[:, 0], pos[:, 1])
        out = rad > 1.0
        if np.any(out):
            pos[out] = pos[out] / rad[out, None]
            vel[out] *= 0.0  # kill velocity for out-of-bounds
        
        it += 1
        
        # Score periodically
        if it % 200 == 0:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                no_improve = 0
            else:
                no_improve += 200
        
        # Early termination if settled
        if it > 5000 and max_f < 0.01:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
            break
    
    final_s = _score(pos)
    if final_s > best_score:
        best_score = final_s
        best_pos = pos.copy()
    
    return best_pos, best_score


def _gradient_refine(pts, start_time, time_limit):
    """Quick gradient ascent on soft-min for final polish."""
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    step = 0.003
    no_improve = 0
    max_no_improve = 300
    it = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break
        
        d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
        np.fill_diagonal(d2, 0.0)
        d = np.sqrt(np.maximum(d2, 0.0))
        np.fill_diagonal(d, np.inf)
        
        rad = np.hypot(pts[:, 0], pts[:, 1])
        r_bound = 1.0 - rad
        min_pair_val = np.min(d) / 2.0
        min_bound_val = np.min(r_bound)
        current_score = min(min_pair_val, min_bound_val)
        
        target = 2.0 * current_score
        pair_violation = np.maximum(0.0, target - d)
        max_v = np.max(pair_violation)
        if max_v > 0:
            exp_v = np.exp(20.0 * (pair_violation - max_v))
            exp_v[~np.isfinite(exp_v)] = 0.0
            sum_exp = np.sum(exp_v)
            pair_weights = exp_v / sum_exp if sum_exp > 0 else np.zeros_like(d)
        else:
            pair_weights = np.zeros_like(d)
        np.fill_diagonal(pair_weights, 0.0)
        
        diff = pts[:, None, :] - pts[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = diff / d[:, :, None]
        unit = np.where(np.isfinite(unit), unit, 0.0)
        grad_pair = -np.sum(pair_weights[:, :, None] * unit, axis=1)
        
        bound_violation = np.maximum(0.0, current_score - r_bound)
        max_bv = np.max(bound_violation)
        if max_bv > 0:
            exp_bv = np.exp(20.0 * (bound_violation - max_bv))
            exp_bv[~np.isfinite(exp_bv)] = 0.0
            sum_exp_bv = np.sum(exp_bv)
            bound_weights = exp_bv / sum_exp_bv if sum_exp_bv > 0 else np.zeros_like(r_bound)
        else:
            bound_weights = np.zeros_like(r_bound)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            inward = -pts / rad[:, None]
        inward = np.where(np.isfinite(inward), inward, 0.0)
        grad_bound = -bound_weights[:, None] * inward
        
        grad = grad_pair + grad_bound - pts * 0.005
        
        fmag = np.linalg.norm(grad, axis=1)
        max_fm = np.max(fmag) if len(fmag) else 1.0
        if max_fm > 0:
            grad = grad * (step / max_fm)
        
        pts = pts + grad
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]
        
        it += 1
        if it % 15 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 15
        
        if it % 100 == 0:
            if no_improve > 150:
                step *= 0.9
            elif no_improve < 30:
                step *= 1.05
            step = min(step, 0.01)
            step = max(step, 0.0001)
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: MD simulation (bulk of the time)
    try:
        md_pts, md_score = _md_simulate(pts, start_time, TIME_LIMIT * 0.65)
        if md_score > best_score:
            best_pts = md_pts
            best_score = md_score
    except Exception:
        pass
    
    # Phase 2: Gradient refinement (remaining time)
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 5.0:
        try:
            grad_pts, grad_score = _gradient_refine(best_pts, start_time, TIME_LIMIT)
            if grad_score > best_score:
                best_pts = grad_pts
                best_score = grad_score
        except Exception:
            pass
    
    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
    
    return best_pts
