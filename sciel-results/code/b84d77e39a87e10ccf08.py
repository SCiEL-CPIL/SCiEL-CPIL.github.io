# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------- geometry helpers ----------
def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts = np.zeros((n, 6, 2))
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _separated(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    # 12 edge normals per hexagon (6 from each hexagon's orientation)
    # For each pair, check all 12 normals (6 from i + 6 from j)
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals = np.stack([np.cos(base), np.sin(base)], axis=2)  # (n, 6, 2)
    
    for i in range(n):
        vi = all_verts[i]  # (6,2)
        ni = hex_normals[i]  # (6,2)
        for j in range(i + 1, n):
            vj = all_verts[j]
            nj = hex_normals[j]
            # Combine all 12 normals
            normals = np.vstack([ni, nj])  # (12, 2)
            proj_i = vi @ normals.T  # (6, 12)
            proj_j = vj @ normals.T  # (6, 12)
            # Check if any normal separates them
            if np.any(np.max(proj_i, axis=0) < np.min(proj_j, axis=0) - tol) \
               or np.any(np.max(proj_j, axis=0) < np.min(proj_i, axis=0) - tol):
                continue
            return False
    return True

def _enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return np.max(projs) / (np.sqrt(3) / 2)

def _soft_max(x, beta=50.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _enclosing_side_soft(positions, angles, beta=50.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return _soft_max(projs, beta) / (np.sqrt(3) / 2)

def _contained(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer[i]
        p2 = outer[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

# ---------- candidate generators ----------
def _sym_candidate(ring_r, valley_r, angle_offset=0.0):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
    for k in range(4):
        a = np.radians(0 + 60 * k + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
    return positions, angles

def _random_candidate(rng):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    angles[0] = rng.uniform(0, 60)
    for k in range(6):
        a = np.radians(30 + 60 * k + rng.uniform(-5, 5))
        r = rng.uniform(1.7, 2.8)
        positions[k + 1] = [r * np.cos(a), r * np.sin(a)]
        angles[k + 1] = rng.uniform(0, 60)
    for k in range(4):
        a = np.radians(0 + 60 * k + rng.uniform(-10, 10))
        r = rng.uniform(2.7, 4.8)
        positions[7 + k] = [r * np.cos(a), r * np.sin(a)]
        angles[7 + k] = rng.uniform(0, 60)
    return positions, angles

def _perturb(positions, angles, rng, scale):
    p = positions + rng.normal(0, scale, positions.shape)
    a = (angles + rng.normal(0, scale * 6, angles.shape)) % 360
    return p, a

# ---------- local refinement ----------
def _refine(positions, angles, deadline, rng):
    best_pos = positions.copy()
    best_ang = angles.copy()
    best_side = _enclosing_side(positions, angles)
    
    def _run(obj, x0, maxiter, maxfev):
        nonlocal best_pos, best_ang, best_side
        try:
            res = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'maxfev': maxfev}
            )
            pos = res.x[:22].reshape(11, 2)
            ang = res.x[22:] % 360
            if _separated(pos, ang):
                s = _enclosing_side(pos, ang)
                if s < best_side:
                    best_side = s
                    best_pos = pos.copy()
                    best_ang = ang.copy()
        except Exception:
            pass
    
    # Phase 1: smooth surrogate with moderate beta
    def obj_smooth(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        if not _separated(pos, ang):
            return 1e6 + 0.001 * np.sum((pos) ** 2)
        return _enclosing_side_soft(pos, ang, beta=30.0)
    
    x0 = np.concatenate([positions.flatten(), angles])
    _run(obj_smooth, x0, 2000, 60000)
    
    # Phase 2: higher beta for sharper peak
    if time.time() < deadline - 30:
        def obj_smooth2(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:]
            if not _separated(pos, ang):
                return 1e6 + 0.001 * np.sum((pos) ** 2)
            return _enclosing_side_soft(pos, ang, beta=100.0)
        
        x1 = np.concatenate([best_pos.flatten(), best_ang])
        _run(obj_smooth2, x1, 1500, 45000)
    
    # Phase 3: exact nonsmooth polish
    if time.time() < deadline - 15:
        def obj_exact(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:]
            if not _separated(pos, ang):
                return 1e6 + 0.001 * np.sum((pos) ** 2)
            return _enclosing_side(pos, ang)
        
        x2 = np.concatenate([best_pos.flatten(), best_ang])
        _run(obj_exact, x2, 3000, 90000)
    
    return best_pos, best_ang, best_side

# ---------- rotation search ----------
def _rotate(positions, angles, theta):
    c = np.cos(np.radians(theta))
    s = np.sin(np.radians(theta))
    rot = np.array([[c, -s], [s, c]])
    return positions @ rot.T, (angles + theta) % 360

def _best_rotation(positions, angles):
    best_t = 0.0
    best_side = _enclosing_side(positions, angles)
    for t in np.linspace(0, 60, 121):
        p, a = _rotate(positions, angles, t)
        s = _enclosing_side(p, a)
        if s < best_side:
            best_side = s
            best_t = t
    try:
        res = minimize(lambda v: _enclosing_side(*_rotate(positions, angles, v[0])),
                       [best_t], method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10})
        if res.fun < best_side:
            best_side = res.fun
            best_t = res.x[0]
    except Exception:
        pass
    return _rotate(positions, angles, best_t), best_side

# ---------- main entry point ----------
def hexagon_packing_11():
    start = time.time()
    deadline = start + 620.0
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_ang = None
    best_side = float('inf')
    
    def consider(pos, ang):
        nonlocal best_pos, best_ang, best_side
        if not _separated(pos, ang):
            return False
        s = _enclosing_side(pos, ang)
        if s < best_side:
            best_side = s
            best_pos = pos.copy()
            best_ang = ang.copy()
            return True
        return False
    
    # --- Phase A: coarse deterministic grid ---
    for ring_r in np.linspace(1.7, 2.7, 14):
        if time.time() > deadline - 180:
            break
        for valley_r in np.linspace(2.8, 4.6, 14):
            if time.time() > deadline - 180:
                break
            for ao in (0.0, 10.0, 20.0, 30.0):
                pos, ang = _sym_candidate(ring_r, valley_r, ao)
                consider(pos, ang)
    
    # --- Phase B: randomized multi-start search + local refinement ---
    n_starts = 0
    while time.time() < deadline - 200:
        if n_starts % 4 == 0:
            pos, ang = _sym_candidate(rng.uniform(1.7, 2.7), rng.uniform(2.8, 4.6), rng.uniform(0, 30))
        elif n_starts % 4 == 1:
            pos, ang = _random_candidate(rng)
        else:
            # Perturb current best
            if best_pos is not None:
                pos, ang = _perturb(best_pos, best_ang, rng, 0.05)
            else:
                pos, ang = _random_candidate(rng)
        n_starts += 1
        consider(pos, ang)
        pos2, ang2, s2 = _refine(pos, ang, deadline, rng)
        if s2 < best_side:
            best_side = s2
            best_pos = pos2.copy()
            best_ang = ang2.copy()
    
    # --- Fallback ---
    if best_pos is None:
        pos, ang = _sym_candidate(2.0, 3.5, 0.0)
        best_pos, best_ang, best_side = pos, ang, _enclosing_side(pos, ang)
    
    # --- Phase C: final polish ---
    if time.time() < deadline - 80:
        pos2, ang2, s2 = _refine(best_pos, best_ang, deadline, rng)
        if s2 < best_side:
            best_pos, best_ang, best_side = pos2, ang2, s2
    
    if time.time() < deadline - 40:
        pos_r, s_r = _best_rotation(best_pos, best_ang)
        if s_r < best_side:
            best_pos, best_side = pos_r, s_r
        if time.time() < deadline - 20:
            pos2, ang2, s2 = _refine(best_pos, best_ang, deadline, rng)
            if s2 < best_side:
                best_pos, best_ang, best_side = pos2, ang2, s2
    
    # --- Final verification ---
    final_side = _enclosing_side(best_pos, best_ang)
    if not _separated(best_pos, best_ang):
        pos, ang = _sym_candidate(2.0, 3.5, 0.0)
        best_pos, best_ang, final_side = pos, ang, _enclosing_side(pos, ang)
    
    if not _contained(best_pos, best_ang, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_pos, best_ang])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(final_side)

# EVOLVE-BLOCK-END
