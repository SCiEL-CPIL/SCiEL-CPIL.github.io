"""
Hexagon packing: minimize outer hexagon side length for N=11 unit hexagons.

Combined strategy:
1. Use smooth log-sum-exp objective for better gradient information.
2. Enumerate polyhexes for good initial topologies.
3. Multi-start local optimization with L-BFGS-B.
4. Final polish with tight tolerances.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY = 10.0

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as list of (x,y) tuples."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...)
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-8):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-7):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-7):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

# ---------------------------------------------------------------------------
# Polyhex enumeration
# ---------------------------------------------------------------------------

def axial_to_cartesian(q, r):
    """Axial hex coordinates -> Cartesian (center of unit hexagon)."""
    x = SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    return (x, y)

def hex_neighbors(q, r):
    """6 neighbors of (q,r) in axial coords."""
    return [(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)]

def _normalize(cells):
    """Translate cells so the lexicographically smallest is at origin."""
    cells = sorted(cells)
    min_q = min(c[0] for c in cells)
    min_r = min(c[1] for c in cells)
    norm = tuple(sorted((q - min_q, r - min_r) for q, r in cells))
    return norm

def _canonical(cells):
    """Canonical form under 6 rotations + 6 reflections of the hex lattice."""
    cells = list(cells)
    best = None
    for rot in range(6):
        for reflect in [False, True]:
            transformed = []
            for q, r in cells:
                if reflect:
                    q, r = r, q
                for _ in range(rot):
                    q, r = -r, q + r
                transformed.append((q, r))
            norm = _normalize(transformed)
            if best is None or norm < best:
                best = norm
    return best

def enumerate_polyhexes(n_cells):
    """Enumerate all polyhexes of size n_cells."""
    if n_cells == 1:
        return [((0, 0),)]
    
    seen = set()
    results = []
    prev = enumerate_polyhexes(n_cells - 1)
    
    for cells in prev:
        cells = list(cells)
        cell_set = set(cells)
        candidates = set()
        for q, r in cells:
            for nq, nr in hex_neighbors(q, r):
                if (nq, nr) not in cell_set:
                    candidates.add((nq, nr))
        
        for cand in candidates:
            new_cells = tuple(sorted(cells + [cand]))
            canon = _canonical(new_cells)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
    
    return results

# ---------------------------------------------------------------------------
# Objective function for optimization
# ---------------------------------------------------------------------------

def objective(params, deadline, smooth_beta=50.0):
    """Objective: smooth outer side length + overlap penalty."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = math.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = math.degrees(params[3 * n + 2]) % 360
    
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, smooth_beta)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            min_d = 2.0 * SQRT3 - 0.05
            if d < min_d:
                excess = min_d - d
                penalty += 200.0 * excess**2
    
    return S + penalty

# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def try_result(params, best_result, best_score, deadline):
    """Try to update best result from params."""
    inner = []
    for i in range(N):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = math.degrees(params[3 * i + 2]) % 360
        inner.append([cx, cy, ang])
    ox = params[3 * N]
    oy = params[3 * N + 1]
    outer_angle = math.degrees(params[3 * N + 2]) % 360
    
    S = compute_required_side(inner, (ox, oy), outer_angle)
    
    if check_validity(inner, (ox, oy), S, outer_angle):
        if S < best_score:
            best_score = S
            best_result = (inner, (ox, oy), S, outer_angle)
    
    return best_result, best_score

def refine_from_params(params, deadline):
    """Refine a set of params using L-BFGS-B. Returns (best_result, best_score)."""
    best_result = None
    best_score = float('inf')
    
    # Try initial params
    best_result, best_score = try_result(params, best_result, best_score, deadline)
    
    # L-BFGS-B refinement with smooth objective
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, params, args=(deadline, 50.0), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < float('inf'):
                best_result, best_score = try_result(res.x, best_result, best_score, deadline)
        except Exception:
            pass
    
    # Second pass with higher beta (sharper max approximation)
    if best_result is not None and time.time() < deadline - 5:
        try:
            res = minimize(objective, params, args=(deadline, 200.0), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < float('inf'):
                best_result, best_score = try_result(res.x, best_result, best_score, deadline)
        except Exception:
            pass
    
    return best_result, best_score

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = START_TIME + TIME_LIMIT - SAFETY
    
    if time.time() > deadline:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_result = None
    best_score = float('inf')
    
    # --- Phase 1: Enumerate polyhexes and get good initial topologies ---
    top_candidates = []
    if time.time() < deadline - 40:
        try:
            polyhexes = enumerate_polyhexes(N)
            scored = []
            for cells in polyhexes:
                if time.time() > deadline - 40:
                    break
                pts = [axial_to_cartesian(q, r) for q, r in cells]
                # Compute a rough bounding size
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                rough_size = max(width, height)
                scored.append((rough_size, cells))
            
            scored.sort(key=lambda x: x[0])
            # Keep top 15 candidates
            K = 15
            top_candidates = [cells for _, cells in scored[:K]]
        except Exception:
            top_candidates = []
    
    # --- Phase 2: Refine top candidates ---
    for cells in top_candidates:
        if time.time() > deadline - 20:
            break
        try:
            # Initial params: axial -> cartesian, all angles 0
            params = np.zeros(3 * N + 3)
            pts = [axial_to_cartesian(q, r) for q, r in cells]
            for i, (px, py) in enumerate(pts):
                params[3 * i] = px
                params[3 * i + 1] = py
                params[3 * i + 2] = 0.0
            # Center at centroid
            cx_init = sum(p[0] for p in pts) / len(pts)
            cy_init = sum(p[1] for p in pts) / len(pts)
            params[3 * N] = cx_init
            params[3 * N + 1] = cy_init
            params[3 * N + 2] = 0.0
            
            result, score = refine_from_params(params, deadline)
            if result is not None:
                if score < best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass
    
    # --- Phase 3: Multi-start with random perturbations ---
    if best_result is not None and time.time() < deadline - 15:
        try:
            # Extract best params
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = math.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = math.radians(best_result[3])
            
            # Try several random perturbations
            np.random.seed(42)
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Add small random noise
                params_perturbed = params_best.copy()
                noise_scale = 0.1 / (attempt + 1)
                for i in range(N):
                    params_perturbed[3 * i] += np.random.uniform(-noise_scale, noise_scale)
                    params_perturbed[3 * i + 1] += np.random.uniform(-noise_scale, noise_scale)
                    params_perturbed[3 * i + 2] += np.random.uniform(-noise_scale, noise_scale)
                params_perturbed[3 * N] += np.random.uniform(-noise_scale, noise_scale)
                params_perturbed[3 * N + 1] += np.random.uniform(-noise_scale, noise_scale)
                params_perturbed[3 * N + 2] += np.random.uniform(-noise_scale, noise_scale)
                
                result, score = refine_from_params(params_perturbed, deadline)
                if result is not None:
                    if score < best_score:
                        best_score = score
                        best_result = result
        except Exception:
            pass
    
    # --- Phase 4: Final polish with very tight tolerances ---
    if best_result is not None and time.time() < deadline - 5:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = math.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = math.radians(best_result[3])
            
            # Use very high beta for near-exact max
            res = minimize(objective, params_best, args=(deadline, 500.0), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < float('inf'):
                best_result, best_score = try_result(res.x, best_result, best_score, deadline)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        # Simple hexagonal arrangement
        hex_positions = [(0.0, 0.0)]
        for k in range(6):
            angle = math.radians(60 * k)
            x = SQRT3 * math.cos(angle)
            y = SQRT3 * math.sin(angle)
            hex_positions.append((x, y))
        for k in range(6):
            angle = math.radians(60 * k)
            x = 2 * SQRT3 * math.cos(angle)
            y = 2 * SQRT3 * math.sin(angle)
            hex_positions.append((x, y))
            if len(hex_positions) >= N:
                break
        hex_positions = hex_positions[:N]
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_required_side(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result


# EVOLVE-BLOCK-END
