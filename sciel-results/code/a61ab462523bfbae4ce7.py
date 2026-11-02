"""Improved program: rigid composite optimization with contact-graph constraint handling."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def hex_vertices_local(angle_rad=0.0):
    """Vertices of unit hexagon centered at origin."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

# Precompute outer normals (for a hexagon with flat-top orientation at angle 0)
# Outer hexagon with side s, centered at (cx,cy), rotated by angle alpha.
# The 6 outward normals are at angles (alpha + 30 + k*60) degrees.
def outer_normals(alpha_rad):
    """6 unit outward normals of the outer hexagon."""
    angs = alpha_rad + np.pi/6.0 + np.arange(6) * np.pi/3.0
    return np.column_stack([np.cos(angs), np.sin(angs)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    alpha = math.radians(outer_angle_deg)
    normals = outer_normals(alpha)
    # For each inner hexagon vertex, compute projection onto each outer normal
    # relative to outer center
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T  # (6, 6)
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    # For a regular hexagon with outward normals n_i, the support function gives
    # the distance from center to each face. The side length s satisfies:
    # face_distance = s * cos(30°) = s * sqrt(3)/2
    # So s = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth (log-sum-exp) surrogate for the max projection."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    alpha = math.radians(outer_angle_deg)
    normals = outer_normals(alpha)
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T
        all_projs.append(projs.ravel())
    P = np.concatenate(all_projs)
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e8:
        return 0.0
    # Use stable log-sum-exp
    shifted = P - m
    # Clip to avoid underflow issues
    shifted = np.clip(shifted, -50.0, 0.0)
    smooth_max = m + math.log(np.sum(np.exp(beta * shifted))) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (numpy arrays (6,2))."""
    # Collect edge normals from both polygons
    axes = []
    for poly in (v1, v2):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def compute_center_distances(inner):
    """Compute pairwise center distances."""
    n = len(inner)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(inner[i][0] - inner[j][0], inner[i][1] - inner[j][1])
            dists[i, j] = dists[j, i] = d
    return dists

def overlap_penalty_smooth(inner, min_dist=SQRT3 - 0.002, beta=30.0):
    """Smooth quadratic penalty when center distances fall below min_dist."""
    n = len(inner)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(inner[i][0] - inner[j][0], inner[i][1] - inner[j][1])
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

# Define the 11-hexagon hexagonal cluster (side-2 hexagon minus corners)
# Using axial coordinates: center + ring of 6 + partial ring
# A regular hexagonal arrangement of 11 hexagons:
# Center (0,0), ring1: 6 hexagons, ring2: 4 hexagons (not full ring of 12)
# Actually, the natural 11-hex cluster is a hexagon of side 2 with 4 corners removed,
# or we use the standard "hex number" arrangement.
# Standard 11: center + 6 in ring 1 + 4 in ring 2 (positions 0,1,3,4 of ring 2)
# Let's use axial coordinates (q, r):
# Ring 0: (0,0)
# Ring 1: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
# Ring 2: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-1,0)... 
# Actually let's just use a known compact 11-hex arrangement:
# A hexagon of "radius" 1.5 doesn't exist. The standard compact 11 is:
# 1 center + 6 around + 4 more = 11
# These form a shape like a larger hexagon with 4 corners missing.

def axial_to_xy(q, r, spacing=1.0):
    """Convert axial coordinates to Cartesian for unit hexagons."""
    x = spacing * 1.5 * q
    y = spacing * SQRT3 * (r + q * 0.5)
    return x, y

def make_lattice_config(spacing=1.0 + 1e-8):
    """Create the 11-hex lattice configuration.
    
    Using a hexagonal number pattern: center + ring1(6) + 4 from ring2.
    The 4 from ring2 are chosen to maintain hexagonal symmetry as much as possible.
    Actually, for 11 hexagons the most symmetric arrangement is:
    - 1 at center
    - 6 in first ring
    - 4 in second ring at alternating positions
    """
    # Axial coordinates for the 11 hexagons
    # Center
    axials = [(0, 0)]
    # Ring 1 (6 hexagons)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    axials.extend(ring1)
    # Ring 2: pick 4 that give good packing
    # Full ring 2 has 12 positions. We pick 4 for symmetry.
    # Positions at angles 0, 60, 180, 240 degrees from center in ring 2
    ring2_full = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
                  (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    # Pick positions 0, 2, 4, 6 (every other) for 4-fold-ish symmetry
    # Actually for hexagonal symmetry, pick 0, 3, 6, 9 -> (2,0), (0,-2), (-2,0), (0,2)
    # That gives a cross/plus shape. Let's try (2,0), (1,-2), (-1,-1), (-2,1) 
    # or just use a different compact set.
    # For best packing, let's use positions that fill the hexagonal shape:
    ring2_pick = [(2, 0), (2, -1), (-2, 1), (-2, 2)]
    axials.extend(ring2_pick)
    
    centers = []
    for q, r in axials:
        x, y = axial_to_xy(q, r, spacing)
        centers.append((x, y))
    return centers

def compute_rigid_composite_side(centers, angles, outer_center, outer_angle_deg):
    """
    Compute the enclosing hexagon side length for a set of unit hexagons
    at given positions and orientations.
    centers: list of (x, y)
    angles: list of angle_degrees
    """
    if not centers:
        return 0.0
    ox, oy = outer_center
    alpha = math.radians(outer_angle_deg)
    normals = outer_normals(alpha)
    max_proj = -np.inf
    for i, ((cx, cy), adeg) in enumerate(zip(centers, angles)):
        verts = hex_vertices(cx, cy, 1.0, math.radians(adeg))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    return 2.0 * max_proj / SQRT3

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin before 150s kill

    # --- Initial configuration: hexagonal lattice ---
    spacing = 1.0 + 1e-7
    init_centers = make_lattice_config(spacing)
    init_angles = [0.0] * N
    init_inner = [[x, y, a] for (x, y), a in zip(init_centers, init_angles)]
    
    cx0 = sum(c[0] for c in init_centers) / N
    cy0 = sum(c[1] for c in init_centers) / N
    
    best_side = compute_required_side_fast(init_inner, (cx0, cy0), 0.0)
    best_inner = [list(c) for c in init_inner]
    best_outer_center = (cx0, cy0)
    best_outer_angle = 0.0
    
    # ============ Phase 1: Optimize outer hexagon only (rigid inner) ============
    # For the lattice, all inner hexagons have same angle (0). 
    # The outer hexagon can be rotated and translated.
    
    def obj_outer(p):
        return compute_required_side_fast(init_inner, (p[0], p[1]), p[2])
    
    best_p1_val = best_side
    best_p1 = np.array([cx0, cy0, 0.0])
    
    # Try multiple initial angles for outer
    for ia in [0, 30, 60, 15, 45]:
        if time.time() > deadline - 90:
            break
        p0 = np.array([cx0, cy0, float(ia)])
        try:
            r = minimize(obj_outer, p0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 4000})
            if r.fun < best_p1_val:
                best_p1_val = r.fun
                best_p1 = r.x.copy()
        except Exception:
            pass
    
    if best_p1_val < best_side:
        best_outer_center = (float(best_p1[0]), float(best_p1[1]))
        best_outer_angle = float(best_p1[2])
        best_side = best_p1_val
    
    # ============ Phase 2: Rigid composite optimization ============
    # Treat all 11 hexagons as one rigid body with 6 DOF:
    # (dx, dy, rotation) for the composite + (ocx, ocy, oangle) for outer
    # But since inner hexagons can have individual angles, we start with all same angle.
    
    def obj_rigid(params):
        """Rigid composite: all inner hexagons share same angle, positioned as lattice."""
        dx, dy, rot, ocx, ocy, oangle = params
        rot_rad = math.radians(rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        inner = []
        for (cx, cy) in init_centers:
            # Rotate position around origin
            nx = cos_r * cx - sin_r * cy + dx
            ny = sin_r * cx + cos_r * cy + dy
            inner.append([nx, ny, rot])
        return compute_required_side_fast(inner, (ocx, ocy), oangle)
    
    best_rigid_val = best_side
    best_rigid_p = None
    p0_rigid = np.array([0.0, 0.0, 0.0, best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    if time.time() < deadline - 80:
        for trial in range(3):
            try:
                if trial == 0:
                    p0 = p0_rigid.copy()
                elif trial == 1:
                    p0 = np.array([0.0, 0.0, 30.0, best_outer_center[0], best_outer_center[1], best_outer_angle])
                else:
                    p0 = np.array([0.0, 0.0, -30.0, best_outer_center[0], best_outer_center[1], best_outer_angle + 15])
                r = minimize(obj_rigid, p0, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 6000})
                if r.fun < best_rigid_val:
                    best_rigid_val = r.fun
                    best_rigid_p = r.x.copy()
            except Exception:
                pass
    
    if best_rigid_p is not None and best_rigid_val < best_side:
        dx, dy, rot, ocx, ocy, oangle = best_rigid_p
        rot_rad = math.radians(rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        new_inner = []
        for (cx, cy) in init_centers:
            nx = cos_r * cx - sin_r * cy + dx
            ny = sin_r * cx + cos_r * cy + dy
            new_inner.append([nx, ny, rot])
        # Validate
        if check_disjoint(new_inner, eps=1e-5):
            best_inner = new_inner
            best_outer_center = (ocx, ocy)
            best_outer_angle = oangle
            best_side = best_rigid_val
    
    # ============ Phase 3: Full joint optimization with smooth objective ============
    # All 36 DOF: 11*(x,y,angle) + 3*(ocx,ocy,oangle)
    
    def make_inner(params):
        inner = []
        for i in range(N):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        return inner
    
    def obj_joint_smooth(params, beta=40.0):
        inner = make_inner(params)
        ocx, ocy, oangle = params[N*3], params[N*3+1], params[N*3+2]
        side = compute_required_side_smooth(inner, (ocx, ocy), oangle, beta=beta)
        # Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(inner[i][0]-inner[j][0], inner[i][1]-inner[j][1])
                if d < SQRT3 - 0.003:
                    pen += (SQRT3 - 0.003 - d) ** 2
        return side + 200.0 * pen
    
    def obj_joint_exact(params):
        inner = make_inner(params)
        ocx, ocy, oangle = params[N*3], params[N*3+1], params[N*3+2]
        if not check_disjoint(inner, eps=1e-4):
            return best_side + 5.0
        return compute_required_side_fast(inner, (ocx, ocy), oangle)
    
    # Build initial params
    p0_joint = np.zeros(N*3 + 3)
    for i, (x, y, a) in enumerate(best_inner):
        p0_joint[i*3] = x
        p0_joint[i*3+1] = y
        p0_joint[i*3+2] = a
    p0_joint[N*3] = best_outer_center[0]
    p0_joint[N*3+1] = best_outer_center[1]
    p0_joint[N*3+2] = best_outer_angle
    
    # Simulated annealing for global search
    def run_sa(p_init, max_iter=5000, T_init=0.5, T_min=0.001):
        current = p_init.copy()
        current_val = obj_joint_smooth(current)
        best_p = current.copy()
        best_val = current_val
        T = T_init
        cool = (T_min / T_init) ** (1.0 / max_iter) if max_iter > 0 else 1.0
        
        for it in range(max_iter):
            if time.time() > deadline - 25:
                break
            nb = current.copy()
            # Perturb
            idx = np.random.randint(0, N*3 + 3)
            if idx < N*3:
                dim = idx % 3
                if dim == 2:  # angle
                    nb[idx] += np.random.normal(0, 3.0)
                else:  # position
                    nb[idx] += np.random.normal(0, 0.15)
            else:  # outer
                dim = (idx - N*3) % 3
                if dim == 2:  # outer angle
                    nb[idx] += np.random.normal(0, 2.0)
                else:  # outer position
                    nb[idx] += np.random.normal(0, 0.10)
            
            nv = obj_joint_smooth(nb)
            if nv < current_val or np.random.random() < math.exp(-(nv - current_val) / max(T, 1e-12)):
                current = nb
                current_val = nv
                if current_val < best_val:
                    best_p = current.copy()
                    best_val = current_val
            T *= cool
        
        return best_p, best_val
    
    np.random.seed(42)
    if time.time() < deadline - 30:
        sa_p, sa_val = run_sa(p0_joint, max_iter=4000)
        # Validate and check exact
        sa_inner = make_inner(sa_p)
        sa_oc = (sa_p[N*3], sa_p[N*3+1])
        sa_oa = sa_p[N*3+2]
        if check_disjoint(sa_inner, eps=1e-5):
            exact_val = compute_required_side_fast(sa_inner, sa_oc, sa_oa)
            if exact_val < best_side:
                best_inner = sa_inner
                best_outer_center = sa_oc
                best_outer_angle = sa_oa
                best_side = exact_val
    
    # ============ Phase 4: Differential Evolution ============
    bounds = []
    for _ in range(N):
        bounds.append((-8.0, 8.0))
        bounds.append((-8.0, 8.0))
        bounds.append((-30.0, 30.0))
    bounds.append((-3.0, 3.0))
    bounds.append((-3.0, 3.0))
    bounds.append((-30.0, 30.0))
    
    def de_obj(params):
        inner = make_inner(params)
        ocx, ocy, oangle = params[N*3], params[N*3+1], params[N*3+2]
        # Quick overlap check using center distances
        for i in range(N):
            for j in range(i+1, N):
                d = math.hypot(inner[i][0]-inner[j][0], inner[i][1]-inner[j][1])
                if d < SQRT3 - 0.01:
                    return best_side + 10.0
        # Full SAT check
        if not check_disjoint(inner, eps=1e-4):
            return best_side + 5.0
        return compute_required_side_fast(inner, (ocx, ocy), oangle)
    
    if time.time() < deadline - 20:
        try:
            de_res = differential_evolution(de_obj, bounds, seed=13, maxiter=50,
                                            popsize=12, tol=1e-7, mutation=(0.4, 1.5),
                                            recombination=0.85, polish=False,
                                            init='sobol', update_interval=5)
            if de_res.fun < best_side:
                di = make_inner(de_res.x)
                doc = (float(de_res.x[N*3]), float(de_res.x[N*3+1]))
                doa = float(de_res.x[N*3+2])
                if check_disjoint(di, eps=1e-5):
                    best_inner = di
                    best_outer_center = doc
                    best_outer_angle = doa
                    best_side = de_res.fun
        except Exception:
            pass
    
    # ============ Phase 5: Final high-precision polish ============
    # Rebuild params from current best
    p0_polish = np.zeros(N*3 + 3)
    for i, (x, y, a) in enumerate(best_inner):
        p0_polish[i*3] = x
        p0_polish[i*3+1] = y
        p0_polish[i*3+2] = a
    p0_polish[N*3] = best_outer_center[0]
    p0_polish[N*3+1] = best_outer_center[1]
    p0_polish[N*3+2] = best_outer_angle
    
    def obj_polish(p):
        inner = make_inner(p)
        ocx, ocy, oangle = p[N*3], p[N*3+1], p[N*3+2]
        if not check_disjoint(inner, eps=1e-5):
            return best_side + 1.0
        return compute_required_side_fast(inner, (ocx, ocy), oangle)
    
    if time.time() < deadline - 3:
        try:
            # Multiple restarts with tiny perturbations
            for trial in range(3):
                if time.time() > deadline - 2:
                    break
                p_init = p0_polish.copy()
                if trial > 0:
                    np.random.seed(trial * 100 + 7)
                    p_init += np.random.normal(0, 0.01, size=p_init.shape)
                r = minimize(obj_polish, p_init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 10000})
                if r.fun < best_side:
                    di = make_inner(r.x)
                    doc = (float(r.x[N*3]), float(r.x[N*3+1]))
                    doa = float(r.x[N*3+2])
                    if check_disjoint(di, eps=1e-6):
                        best_inner = di
                        best_outer_center = doc
                        best_outer_angle = doa
                        best_side = r.fun
        except Exception:
            pass
    
    # Final validation
    if not check_disjoint(best_inner, eps=1e-4):
        # Fall back to initial lattice
        best_inner = [list(c) for c in init_inner]
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    # Ensure angles are in reasonable range
    for i in range(N):
        best_inner[i][2] = best_inner[i][2] % 60.0  # Hexagon symmetry
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
