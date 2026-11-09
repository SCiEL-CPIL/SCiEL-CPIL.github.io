"""
Hexagon packing: place N=11 unit regular hexagons inside a regular hexagon,
minimizing the outer side length. Each inner hexagon may translate and rotate.

Strategy:
- Simulated annealing with a smooth energy function (overlap + containment penalties).
- Force-directed refinement using SAT-based penetration depths for smooth gradients.
- Multi-start with diverse initial topologies.
- Final validity check and minimal side computation.
"""

import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _sat_penetration_depth(verts1, verts2):
    """
    Compute the minimum penetration depth between two convex hexagons using SAT.
    Returns (depth, normal) where depth > 0 means overlap, and normal is the
    direction to separate them (from 1 to 2). Returns (0, None) if disjoint.
    """
    n1 = len(verts1)
    n2 = len(verts2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_overlap = float('inf')
    best_normal = None
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0, None  # Disjoint
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            best_normal = np.array([nx, ny])
    
    if best_normal is None:
        return 0.0, None
    
    # Ensure normal points from 1 to 2
    c1 = np.mean(verts1, axis=0)
    c2 = np.mean(verts2, axis=0)
    if np.dot(best_normal, c2 - c1) < 0:
        best_normal = -best_normal
    
    return min_overlap, best_normal

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            depth, _ = _sat_penetration_depth(all_verts[i], all_verts[j])
            if depth > tol:
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement with correct spacing."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)  # Minimum distance for non-overlapping unit hexagons
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    # Add 4 more hexagons in a second ring
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _initial_circular():
    """Create a circular initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        a = 2 * np.pi * i / N
        r = 2.0 * np.sqrt(3)
        positions.append([r * np.cos(a), r * np.sin(a), 0.0])
    return positions

def _compute_energy(params, outer_side):
    """
    Compute energy: outer_side + overlap_penalty + containment_penalty.
    params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    """
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3 + 1]
        a = params[i*3 + 2]
        inner.append((x, y, a))
    
    cx = params[N*3]
    cy = params[N*3 + 1]
    cangle = params[N*3 + 2]
    
    # Compute actual required side
    actual_side = _min_enclosing_side(inner, (cx, cy), cangle)
    
    # Overlap penalty
    all_verts = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            depth, _ = _sat_penetration_depth(all_verts[i], all_verts[j])
            if depth > 0:
                overlap_penalty += depth ** 2
    
    # Containment penalty: how much the hexagons stick out
    containment_penalty = 0.0
    for (ix, iy, iangle) in inner:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, cangle, tol=0):
                # Compute distance outside
                dx = v[0] - cx
                dy = v[1] - cy
                angle_rad = np.radians(cangle)
                ca = np.cos(-angle_rad)
                sa = np.sin(-angle_rad)
                lx = dx * ca - dy * sa
                ly = dx * sa + dy * ca
                
                inradius = outer_side * np.sqrt(3) / 2
                max_violation = 0.0
                for k in range(6):
                    a = np.radians(30 + k * 60)
                    nx = np.cos(a)
                    ny = np.sin(a)
                    val = lx * nx + ly * ny - inradius
                    if val > max_violation:
                        max_violation = val
                containment_penalty += max_violation ** 2
    
    return actual_side + overlap_penalty * 10.0 + containment_penalty * 10.0

def _simulate_annealing(initial_positions, outer_angle=0.0, num_iterations=20000):
    """Simulated annealing to minimize energy."""
    # Initialize
    params = []
    for p in initial_positions:
        params.extend(p)
    params.extend([0.0, 0.0, outer_angle])
    
    current_energy = _compute_energy(params, 5.0)
    best_params = params.copy()
    best_energy = current_energy
    
    T = 0.5
    T_min = 0.001
    cooling = (T_min / T) ** (1.0 / num_iterations)
    
    for it in range(num_iterations):
        if _remaining_time() < 5.0:
            break
        
        # Random perturbation
        new_params = params.copy()
        idx = random.randint(0, N*3 - 1)
        if idx % 3 == 2:  # angle
            new_params[idx] += random.gauss(0, 5.0)
            new_params[idx] = new_params[idx] % 360
        else:  # x or y
            new_params[idx] += random.gauss(0, 0.5)
        
        new_energy = _compute_energy(new_params, 5.0)
        
        # Accept or reject
        if new_energy < current_energy:
            params = new_params
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()
        else:
            delta = new_energy - current_energy
            if random.random() < math.exp(-delta / T):
                params = new_params
                current_energy = new_energy
        
        T *= cooling
    
    return best_params, best_energy

def _gradient_refine(params, outer_angle=0.0, max_iters=500):
    """Use L-BFGS-B to refine the configuration."""
    # Re-center
    xs = [params[i*3] for i in range(N)]
    ys = [params[i*3+1] for i in range(N)]
    cx_init = sum(xs) / len(xs)
    cy_init = sum(ys) / len(ys)
    
    shifted = params.copy()
    for i in range(N):
        shifted[i*3] -= cx_init
        shifted[i*3+1] -= cy_init
    shifted[N*3] = 0.0
    shifted[N*3+1] = 0.0
    shifted[N*3+2] = outer_angle
    
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((0.0, 60.0))
    bounds.append((-5.0, 5.0))
    bounds.append((-5.0, 5.0))
    bounds.append((0.0, 60.0))
    
    def obj(p):
        if _remaining_time() < 3.0:
            return 1e10
        return _compute_energy(p, 5.0)
    
    try:
        result = minimize(obj, shifted, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iters, 'maxfun': max_iters*5})
        return result.x
    except Exception:
        return shifted

def _try_configuration(initial_positions, outer_angle=0.0):
    """Try a full optimization pipeline from an initial configuration."""
    # Simulated annealing
    sa_params, sa_energy = _simulate_annealing(initial_positions, outer_angle, num_iterations=10000)
    
    # Gradient refinement
    refined_params = _gradient_refine(sa_params, outer_angle, max_iters=300)
    
    # Extract configuration
    inner = []
    for i in range(N):
        x = refined_params[i*3]
        y = refined_params[i*3+1]
        a = refined_params[i*3+2] % 360
        inner.append([x, y, a])
    
    cx = refined_params[N*3]
    cy = refined_params[N*3+1]
    cangle = refined_params[N*3+2] % 60
    
    side = _min_enclosing_side(inner, (cx, cy), cangle)
    
    # Verify validity
    if _check_validity(inner, (cx, cy), side, cangle, tol=1e-5):
        return inner, (cx, cy), side, cangle
    else:
        # Try with slightly larger side
        side += 0.05
        if _check_validity(inner, (cx, cy), side, cangle, tol=1e-5):
            return inner, (cx, cy), side, cangle
        else:
            return None

def construct():
    start_time = time.time()
    
    initializations = [
        _initial_lattice,
        _initial_circular,
        _initial_random
    ]
    
    best_result = None
    best_side = float('inf')
    
    for init_func in initializations:
        if _remaining_time() < 15.0:
            break
        
        init_pos = init_func()
        
        for angle in [0.0, 30.0]:
            if _remaining_time() < 15.0:
                break
            
            result = _try_configuration(init_pos, outer_angle=angle)
            
            if result:
                inner, center, side, cangle = result
                if side < best_side:
                    best_side = side
                    best_result = result
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0
        return inner, center, side, angle
    
    inner, center, side, angle = best_result
    
    # Final verification
    if not _check_validity(inner, center, side, angle, tol=1e-5):
        side += 0.1
        if _check_validity(inner, center, side, angle, tol=1e-5):
            return inner, center, side, angle
        else:
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            center = (0.0, 0.0)
            side = 40.0
            angle = 0.0
            return inner, center, side, angle
    
    return inner, center, side, angle
