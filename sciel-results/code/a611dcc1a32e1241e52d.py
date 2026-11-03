import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    base = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180.0
    angles = base + angle_rad
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def _poly_overlap_sat(poly1, poly2):
    """SAT test: return True if two convex polygons overlap (interiors intersect)."""
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Edge normal
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found
    return True  # No separating axis -> overlap

def _check_nonoverlap(centers, angles_deg, tol=1e-9):
    """Check all pairs for non-overlap. Returns True if valid."""
    polys = [_hex_vertices(centers[i][0], centers[i][1], angles_deg[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_overlap_sat(polys[i], polys[j]):
                return False
    return True

def _min_outer_side(centers, angles_deg, outer_angle_deg=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    theta = np.deg2rad(outer_angle_deg)
    # 6 normals of outer hexagon (pointing outward from center)
    normals = np.array([[np.cos(theta + np.deg2rad(30 + k * 60)),
                         np.sin(theta + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles_deg[i])
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    
    # For a regular hexagon with side S, the apothem (inradius) = S * sqrt(3)/2
    # max_proj is the max distance from center to any vertex along any normal
    # S = 2 * max_proj / sqrt(3)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _objective_full(x):
    """Objective: outer side length + large penalty for overlaps.
    x = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, outer_angle] all in radians for angles."""
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    outer_ang_deg = np.rad2deg(x[-1])
    
    S = _min_outer_side(centers, angles_deg, outer_ang_deg)
    
    # Check overlaps
    polys = [_hex_vertices(centers[i][0], centers[i][1], angles_deg[i]) for i in range(n)]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Quick distance check first
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < 2.0:  # Only check if potentially overlapping
                if _poly_overlap_sat(polys[i], polys[j]):
                    # Estimate overlap depth
                    penalty += max(0, 2.0 - d)
    return S + 100.0 * penalty

def _init_ring_topology(rot_deg=0.0):
    """Center + 6 ring + 4 outer shell. Returns (centers, angles_deg)."""
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Ring topology: center, 6 neighbors, 4 from shell 2
    lattice = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
               (2,0), (1,1), (0,2), (-2,0)]
    
    centers = []
    for i, j in lattice:
        c = i * A + j * B
        centers.append(tuple(c))
    
    # Rotate
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in centers]
    
    angles_deg = [0.0] * N
    return rotated, angles_deg

def _init_staggered(rot_deg=0.0):
    """Alternative: staggered arrangement."""
    # Two rows of 5+6 or similar
    centers = []
    # Row 1: 6 hexagons
    for k in range(6):
        x = k * np.sqrt(3) - 2.5 * np.sqrt(3)
        y = 0.0
        centers.append((x, y))
    # Row 2: 5 hexagons offset
    for k in range(5):
        x = k * np.sqrt(3) - 2.0 * np.sqrt(3) + np.sqrt(3)/2
        y = 1.5
        centers.append((x, y))
    
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in centers]
    
    angles_deg = [0.0] * N
    return rotated, angles_deg

def _init_compact(rot_deg=0.0):
    """Tighter arrangement: 3-4-3-1 or similar."""
    centers = []
    # Try a more compact layout
    positions = [
        (0, 0),
        (np.sqrt(3), 0), (-np.sqrt(3), 0),
        (np.sqrt(3)/2, 1.5), (-np.sqrt(3)/2, 1.5),
        (np.sqrt(3)/2, -1.5), (-np.sqrt(3)/2, -1.5),
        (2*np.sqrt(3), 0), (-2*np.sqrt(3), 0),
        (np.sqrt(3), 3.0), (-np.sqrt(3), 3.0),
        (0, -3.0),
    ]
    centers = positions[:N]
    
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in centers]
    
    angles_deg = [0.0] * N
    return rotated, angles_deg

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_S = float('inf')
    best_config = None
    
    # Generate diverse initial configurations
    init_funcs = [_init_ring_topology, _init_staggered, _init_compact]
    rotations = [0, 30, 60, 90, 120, 150, 210, 240, 300, 330]
    inner_angles = [0, 30, 60]
    
    initial_configs = []
    for rot in rotations:
        for func in init_funcs:
            centers, angles = func(rot)
            for ia in inner_angles:
                angles_rot = [ia] * N
                initial_configs.append((centers, angles_rot, 0.0))
    
    # Phase 1: Evaluate all initial configs
    for centers, angles_deg, outer_ang in initial_configs:
        if time.time() > deadline - 100:
            break
        S = _min_outer_side(centers, angles_deg, outer_ang)
        if _check_nonoverlap(centers, angles_deg) and S < best_S:
            best_S = S
            best_config = (centers, angles_deg, outer_ang)
    
    if best_config is None:
        # Fallback
        centers, angles = _init_ring_topology(0)
        best_config = (centers, angles, 0.0)
        best_S = _min_outer_side(centers, angles, 0.0)
    
    # Phase 2: Nelder-Mead refinement from best and nearby configs
    def _run_nm(centers, angles_deg, outer_ang, maxiter=200):
        x0 = []
        for (cx, cy), ang in zip(centers, angles_deg):
            x0.extend([cx, cy, np.deg2rad(ang)])
        x0.append(np.deg2rad(outer_ang))
        
        result = minimize(
            _objective_full,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        return result
    
    # Refine best config
    if time.time() < deadline - 80:
        centers, angles_deg, outer_ang = best_config
        try:
            res = _run_nm(centers, angles_deg, outer_ang, maxiter=300)
            if res.fun < best_S + 1e-6:
                x_opt = res.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                if _check_nonoverlap(opt_centers, opt_angles):
                    S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                    if S < best_S:
                        best_S = S
                        best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 3: Multi-start NM from perturbed best and other good configs
    if time.time() < deadline - 60:
        # Try several other initial configs with NM
        tried = 0
        for centers, angles_deg, outer_ang in initial_configs:
            if time.time() > deadline - 55:
                break
            if tried >= 8:
                break
            S_init = _min_outer_side(centers, angles_deg, outer_ang)
            if S_init < best_S + 0.5:  # Only try promising ones
                try:
                    res = _run_nm(centers, angles_deg, outer_ang, maxiter=150)
                    if res.fun < best_S + 1e-6:
                        x_opt = res.x
                        opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                        opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                        opt_outer = np.rad2deg(x_opt[-1])
                        if _check_nonoverlap(opt_centers, opt_angles):
                            S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                            if S < best_S:
                                best_S = S
                                best_config = (opt_centers, opt_angles, opt_outer)
                    tried += 1
                except Exception:
                    pass
    
    # Phase 4: L-BFGS-B from best
    if time.time() < deadline - 30:
        centers, angles_deg, outer_ang = best_config
        x0 = []
        for (cx, cy), ang in zip(centers, angles_deg):
            x0.extend([cx, cy, np.deg2rad(ang)])
        x0.append(np.deg2rad(outer_ang))
        
        try:
            lbfgs = minimize(
                _objective_full,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if lbfgs.fun < best_S + 1e-6:
                x_opt = lbfgs.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                if _check_nonoverlap(opt_centers, opt_angles):
                    S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                    if S < best_S:
                        best_S = S
                        best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 5: Final NM polish
    if time.time() < deadline - 15:
        centers, angles_deg, outer_ang = best_config
        try:
            res = _run_nm(centers, angles_deg, outer_ang, maxiter=400)
            if res.fun < best_S + 1e-6:
                x_opt = res.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                if _check_nonoverlap(opt_centers, opt_angles):
                    S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                    if S < best_S:
                        best_S = S
                        best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 6: Small perturbations + NM
    if time.time() < deadline - 5:
        centers, angles_deg, outer_ang = best_config
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            x_pert = []
            for (cx, cy), ang in zip(centers, angles_deg):
                x_pert.extend([
                    cx + np.random.uniform(-0.05, 0.05),
                    cy + np.random.uniform(-0.05, 0.05),
                    np.deg2rad(ang) + np.random.uniform(-0.1, 0.1)
                ])
            x_pert.append(np.deg2rad(outer_ang) + np.random.uniform(-0.1, 0.1))
            
            try:
                res = minimize(
                    _objective_full,
                    x_pert,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if res.fun < best_S + 1e-6:
                    x_opt = res.x
                    opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(x_opt[-1])
                    if _check_nonoverlap(opt_centers, opt_angles):
                        S = _min_outer_side(opt_centers, opt_angles, opt_outer)
                        if S < best_S:
                            best_S = S
                            best_config = (opt_centers, opt_angles, opt_outer)
            except Exception:
                pass
    
    centers, angles_deg, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles_deg)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
