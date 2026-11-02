"""
Optimized packing of 12 unit regular hexagons inside a regular hexagon.
Uses a honeycomb initial configuration, fast SAT with precomputed normals,
adaptive simulated annealing, and a two-pass Nelder-Mead polish.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 130.0  # Safety margin before 150s wall-clock kill

# Precomputed constants
S3 = math.sqrt(3)
COS60 = 0.5
SIN60 = S3 / 2.0
PI = math.pi

# Base hexagon vertices (unit side, flat-top orientation, angle=0)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
BASE_VERTS = np.array([
    [1.0, 0.0],
    [COS60, SIN60],
    [-COS60, SIN60],
    [-1.0, 0.0],
    [-COS60, -SIN60],
    [COS60, -SIN60]
])

# Precompute edge normals for the base hexagon (outward-facing)
# For edge from v[i] to v[i+1], normal is perpendicular to edge
EDGE_NORMALS = np.zeros((6, 2))
for i in range(6):
    e = BASE_VERTS[(i+1)%6] - BASE_VERTS[i]
    # Outward normal: rotate edge by -90 degrees (clockwise)
    EDGE_NORMALS[i] = np.array([e[1], -e[0]])
    EDGE_NORMALS[i] /= np.linalg.norm(EDGE_NORMALS[i])

def get_hex_vertices(cx, cy, angle_deg):
    """Fast computation of hexagon vertices using precomputed base vertices."""
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    return cx + cy * 0 + (BASE_VERTS @ rot.T) + np.array([cx, cy])

def get_hex_vertices_fast(cx, cy, angle_deg):
    """Even faster: manual rotation without matrix multiply."""
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    v = np.empty((6, 2))
    for i in range(6):
        bx = BASE_VERTS[i, 0]
        by = BASE_VERTS[i, 1]
        v[i, 0] = cx + bx * ca - by * sa
        v[i, 1] = cy + bx * sa + by * ca
    return v

def hexagons_disjoint_fast(h1, h2):
    """
    Fast SAT check for two hexagons.
    h1, h2 are (x, y, angle_deg).
    Returns True if disjoint.
    """
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx*dx + dy*dy
    
    # If far apart, definitely disjoint (max radius is 1, so dist > 2 means disjoint)
    if dist_sq > 4.0:
        return True
    
    # Get vertices
    v1 = get_hex_vertices_fast(h1[0], h1[1], h1[2])
    v2 = get_hex_vertices_fast(h2[0], h2[1], h2[2])
    
    # SAT: Check 6 normals from each hexagon
    # For hexagon 1, normals are rotated versions of EDGE_NORMALS
    a1 = math.radians(h1[2])
    ca1 = math.cos(a1)
    sa1 = math.sin(a1)
    
    a2 = math.radians(h2[2])
    ca2 = math.cos(a2)
    sa2 = math.sin(a2)
    
    # Check normals from hexagon 1
    for i in range(6):
        nx = EDGE_NORMALS[i, 0] * ca1 - EDGE_NORMALS[i, 1] * sa1
        ny = EDGE_NORMALS[i, 0] * sa1 + EDGE_NORMALS[i, 1] * ca1
        
        proj1 = v1[:, 0]*nx + v1[:, 1]*ny
        proj2 = v2[:, 0]*nx + v2[:, 1]*ny
        
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return True
    
    # Check normals from hexagon 2
    for i in range(6):
        nx = EDGE_NORMALS[i, 0] * ca2 - EDGE_NORMALS[i, 1] * sa2
        ny = EDGE_NORMALS[i, 0] * sa2 + EDGE_NORMALS[i, 1] * ca2
        
        proj1 = v1[:, 0]*nx + v1[:, 1]*ny
        proj2 = v2[:, 0]*nx + v2[:, 1]*ny
        
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return True
    
    return False

def is_valid_configuration(inner):
    """Check all pairs for disjointness."""
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                return False
    return True

def compute_min_outer_side(inner):
    """
    Compute the minimum side length of a regular hexagon containing all inner hexagons.
    Optimizes the rotation angle of the outer hexagon.
    """
    # Get all vertices of inner hexagons
    all_verts = np.vstack([get_hex_vertices_fast(h[0], h[1], h[2]) for h in inner])
    
    # Compute center: use centroid of convex hull vertices
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
        center = np.mean(hull_verts, axis=0)
    except Exception:
        center = np.mean(all_verts, axis=0)
    
    offsets = all_verts - center
    
    # Find optimal rotation angle phi for outer hexagon
    # Outer hexagon face normals are at angles phi + 30 + k*60 degrees
    # Required side S = max_k (2/sqrt(3) * max_v |n_k . offset_v|)
    
    sqrt3 = S3
    best_side = float('inf')
    best_phi = 0.0
    
    # Coarse scan: 0 to 60 degrees in steps of 1 degree
    for phi_deg in range(0, 60):
        phi = np.radians(phi_deg)
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi + PI/6 + k * PI/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            side_k = (2.0 / sqrt3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
        if max_extent < best_side:
            best_side = max_extent
            best_phi = phi_deg
    
    # Fine scan around best
    for phi_deg in np.arange(best_phi - 1, best_phi + 1, 0.05):
        if phi_deg < 0 or phi_deg >= 60:
            continue
        phi = np.radians(phi_deg)
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi + PI/6 + k * PI/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            side_k = (2.0 / sqrt3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
        if max_extent < best_side:
            best_side = max_extent
            best_phi = phi_deg
    
    # Polish with Nelder-Mead
    def obj_phi(phi_rad):
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi_rad + PI/6 + k * PI/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            side_k = (2.0 / sqrt3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
        return max_extent
    
    phi_init = np.radians(best_phi)
    for start_offset in [0, 0.05, -0.05, 0.1, -0.1]:
        start_phi = phi_init + start_offset
        try:
            res = minimize(obj_phi, start_phi, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
            if res.fun < best_side:
                best_side = res.fun
                best_phi = np.degrees(res.x[0]) % 60.0
        except:
            pass
    
    return best_side, best_phi, center

def make_honeycomb_initial():
    """
    Create a valid initial packing of 12 hexagons.
    Uses a grid with sufficient spacing to ensure disjointness.
    """
    inner = []
    # Spacing 2.0 ensures horizontal neighbors touch at vertices (disjoint).
    # Spacing 1.8 ensures vertical neighbors are disjoint (height is sqrt(3) ~ 1.732).
    h_spacing = 2.0
    v_spacing = 1.8
    
    # Row 0: 3 hexagons
    for c in range(3):
        x = (c - 1) * h_spacing
        y = 0.0
        inner.append([x, y, 0.0])
    # Row 1: 4 hexagons
    for c in range(4):
        x = (c - 1.5) * h_spacing
        y = v_spacing
        inner.append([x, y, 0.0])
    # Row 2: 3 hexagons
    for c in range(3):
        x = (c - 1) * h_spacing
        y = 2 * v_spacing
        inner.append([x, y, 0.0])
    # Row 3: 2 hexagons
    for c in range(2):
        x = (c - 0.5) * h_spacing
        y = 3 * v_spacing
        inner.append([x, y, 0.0])
    
    return inner

def construct():
    t0 = time.time()
    
    # Initial Configuration: Honeycomb packing
    initial_inner = make_honeycomb_initial()
    
    best_inner = [list(h) for h in initial_inner]
    best_side, best_phi, best_center = compute_min_outer_side(best_inner)
    
    current_inner = [list(h) for h in initial_inner]
    current_side, _, _ = compute_min_outer_side(current_inner)
    
    # Simulated Annealing with adaptive step size
    T = 0.5
    T_min = 0.005
    cooling = 0.9995
    step_size = 0.3
    min_step = 0.01
    max_steps = 30000
    
    step_count = 0
    accept_count = 0
    improve_count = 0
    
    # Check if initial configuration is valid
    current_valid = is_valid_configuration(current_inner)
    if not current_valid:
        current_side = 100.0  # Force acceptance of any valid move

    while time.time() - t0 < DEADLINE and step_count < max_steps:
        step_count += 1
        
        # Adaptive step size
        if step_count % 100 == 0:
            if accept_count > 50:
                step_size *= 0.95
            elif accept_count < 10:
                step_size *= 1.1
            step_size = max(min_step, min(step_size, 1.0))
            accept_count = 0
        
        # Perturb one random hexagon
        idx = random.randint(0, N - 1)
        new_inner = [list(h) for h in current_inner]
        
        # Perturb position and angle
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 5.0)  # degrees
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        # Check validity (disjointness) - only check moved hexagon against others
        valid = True
        for j in range(N):
            if j == idx:
                continue
            if not hexagons_disjoint_fast(new_inner[idx], new_inner[j]):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min:
                T = T_min
            continue
        
        # Compute new side length
        new_side, new_phi, new_center = compute_min_outer_side(new_inner)
        
        # Accept if better or with probability exp(-delta/T)
        if new_side < current_side:
            current_inner = new_inner
            current_side = new_side
            current_valid = True
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(h) for h in current_inner]
                best_phi = new_phi
                best_center = new_center
                improve_count += 1
        else:
            delta = new_side - current_side
            if delta < 0:
                delta = 0
            if delta < 5 * T and random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_side = new_side
                current_valid = True
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_inner = [list(h) for h in current_inner]
                    best_phi = new_phi
                    best_center = new_center
                    improve_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final Polish: Nelder-Mead on all positions simultaneously
    def flatten_inner(inner):
        return np.array([item for sublist in inner for item in sublist])
    
    def unflatten_inner(flat):
        inner = []
        for i in range(N):
            inner.append([flat[3*i], flat[3*i+1], flat[3*i+2]])
        return inner
    
    def objective(flat):
        inner = unflatten_inner(flat)
        
        try:
            side, phi, center = compute_min_outer_side(inner)
        except:
            return 1e10
        
        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < S3:
                    penalty += (S3 - dist)**2
        
        return side + 50.0 * penalty
    
    current_flat = flatten_inner(best_inner)
    
    time_left = DEADLINE - (time.time() - t0)
    if time_left > 8.0:
        try:
            # First pass: moderate iterations
            res = minimize(objective, current_flat, method='Nelder-Mead',
                          options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300, 'maxfev': 800})
            
            if res.fun < best_side + 0.01:
                polished_inner = unflatten_inner(res.x)
                # Verify validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint_fast(polished_inner[i], polished_inner[j]):
                            valid = False
                            break
                        if j == i:
                            continue
                if valid:
                    new_side, new_phi, new_center = compute_min_outer_side(polished_inner)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = polished_inner
                        best_phi = new_phi
                        best_center = new_center
                        
                        # Second pass: tighter tolerance
                        time_left2 = DEADLINE - (time.time() - t0)
                        if time_left2 > 5.0:
                            try:
                                res2 = minimize(objective, flatten_inner(best_inner), method='Nelder-Mead',
                                              options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 1500})
                                if res2.fun < best_side + 0.005:
                                    polished2 = unflatten_inner(res2.x)
                                    valid2 = True
                                    for i in range(N):
                                        for j in range(i+1, N):
                                            if not hexagons_disjoint_fast(polished2[i], polished2[j]):
                                                valid2 = False
                                                break
                                    if valid2:
                                        s2, p2, c2 = compute_min_outer_side(polished2)
                                        if s2 < best_side:
                                            best_side = s2
                                            best_inner = polished2
                                            best_phi = p2
                                            best_center = c2
                            except:
                                pass
        except:
            pass
    
    # Final Result
    return best_inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_phi)
