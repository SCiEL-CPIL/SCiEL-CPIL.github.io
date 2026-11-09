"""Hexagon packing: 11 unit hexagons in a regular hexagon.
Uses a different modelling family: represent each inner hexagon by 6 vertices,
optimize with SLSQP using smooth penetration depth as constraint, and compute
the outer side analytically from vertex containment.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize, LinearConstraint

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

def _hex_edge_normals(verts):
    """Return unit outward normals for each edge of a convex polygon (vertices CCW)."""
    n = len(verts)
    normals = []
    for i in range(n):
        j = (i + 1) % n
        edge = verts[j] - verts[i]
        length = np.linalg.norm(edge)
        if length < 1e-12:
            normals.append(np.array([1.0, 0.0]))
        else:
            nx = -edge[1] / length
            ny = edge[0] / length
            normals.append(np.array([nx, ny]))
    return np.array(normals)

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

def _penetration_depth(verts1, verts2):
    """Compute minimum penetration depth between two convex hexagons using SAT.
    Returns 0 if separated, positive if overlapping."""
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    max_overlap = 0.0
    for nx, ny in all_normals:
        proj1 = verts1 @ np.array([nx, ny])
        proj2 = verts2 @ np.array([nx, ny])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0  # Separated
        
        overlap = min(max1, max2) - max(min1, min2)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    # Check all inner hexagons are inside outer
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
            
            inradius = outer_side * np.sqrt(3) / 2
            for k in range(6):
                a = np.radians(30 + k * 60)
                nx = np.cos(a)
                ny = np.sin(a)
                if lx * nx + ly * ny > inradius + tol:
                    return False
    
    # Check no overlaps between inner hexagons
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if _penetration_depth(all_verts[i], all_verts[j]) > tol:
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_ring():
    """Create a ring arrangement with center."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    r = 2.0 * np.sqrt(3)
    for k in range(10):
        a = 2 * np.pi * k / 10
        positions.append([r * np.cos(a), r * np.sin(a), 0.0])
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.5, 4.5)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _optimize_slsqp(positions, outer_angle, max_iter=300):
    """Optimize using SLSQP with smooth penetration constraints."""
    x0 = np.array([p for pos in positions for p in pos])
    
    def objective(x):
        pos = x.reshape(N, 3)
        side = _min_enclosing_side([tuple(p) for p in pos], (0.0, 0.0), outer_angle)
        return side
    
    # Constraints: no overlap between any pair
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def constraint(x):
                    pos = x.reshape(N, 3)
                    v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, pos[i][2])
                    v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, pos[j][2])
                    # Want penetration <= 0, so constraint is -penetration >= 0
                    return -_penetration_depth(v1, v2)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = [(-20, 20)] * (N * 2) + [(0, 360)] * N
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-8}
    )
    
    best_x = result.x.reshape(N, 3)
    best_x[:, 2] = best_x[:, 2] % 360
    
    return [tuple(p) for p in best_x], result.fun

def _optimize_lbfgs(positions, outer_angle, max_iter=500):
    """Optimize using L-BFGS-B with penalty method."""
    x0 = np.array([p for pos in positions for p in pos])
    
    def objective(x):
        pos = x.reshape(N, 3)
        side = _min_enclosing_side([tuple(p) for p in pos], (0.0, 0.0), outer_angle)
        
        # Penalty for overlaps
        penalty = 0.0
        verts_list = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in pos]
        for i in range(N):
            for j in range(i+1, N):
                pd = _penetration_depth(verts_list[i], verts_list[j])
                if pd > 0:
                    penalty += pd * pd
        
        return side + 50.0 * penalty
    
    bounds = [(-20, 20)] * (N * 2) + [(0, 360)] * N
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-10}
    )
    
    best_x = result.x.reshape(N, 3)
    best_x[:, 2] = best_x[:, 2] % 360
    
    return [tuple(p) for p in best_x], result.fun

def _try_arrangement(positions, outer_angle):
    """Try to optimize a given arrangement and return the best result."""
    if _remaining_time() < 10.0:
        return None, float('inf')
    
    best_positions = None
    best_side = float('inf')
    
    # Try SLSQP first
    try:
        opt_positions, opt_value = _optimize_slsqp(positions, outer_angle, max_iter=200)
        side = _min_enclosing_side(opt_positions, (0.0, 0.0), outer_angle)
        if _check_validity(opt_positions, (0.0, 0.0), side, outer_angle, tol=1e-5):
            best_positions = opt_positions
            best_side = side
    except Exception:
        pass
    
    # Try L-BFGS-B
    if _remaining_time() > 15.0:
        try:
            opt_positions, opt_value = _optimize_lbfgs(positions, outer_angle, max_iter=300)
            side = _min_enclosing_side(opt_positions, (0.0, 0.0), outer_angle)
            if _check_validity(opt_positions, (0.0, 0.0), side, outer_angle, tol=1e-5):
                if side < best_side:
                    best_positions = opt_positions
                    best_side = side
        except Exception:
            pass
    
    if best_positions is not None:
        return best_positions, best_side
    
    return None, float('inf')

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0]
    
    initial_configs = [
        _initial_lattice(),
        _initial_ring(),
        _initial_random(),
        _initial_random(),
        _initial_random(),
    ]
    
    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break
        
        for init_pos in initial_configs:
            if _remaining_time() < 15.0:
                break
            
            positions = [list(p) for p in init_pos]
            
            result, side = _try_arrangement(positions, oa)
            
            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
