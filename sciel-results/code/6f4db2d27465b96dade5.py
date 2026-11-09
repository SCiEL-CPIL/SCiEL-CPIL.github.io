"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Multi-start Nelder-Mead Simplex optimization.
Directly minimizes the outer side length (max support function) with a large
penalty for violations (overlap and containment).
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
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
                # Calculate angular distance to nearest face normal
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

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
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
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

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
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
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
    angles2 = [30, 90, 150, 210, 270, 330]
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

def _objective(x, outer_angle):
    """
    Objective function for Nelder-Mead.
    x: flattened array of [x1, y1, a1, x2, y2, a2, ...]
    Returns: side_length + penalty
    """
    positions = []
    for i in range(N):
        positions.append([x[i*3], x[i*3+1], x[i*3+2]])
    
    # 1. Calculate required side length
    side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    
    # 2. Calculate penalties
    penalty = 0.0
    all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in positions]
    
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-6):
                # Estimate overlap depth roughly by center distance
                ci = np.mean(all_verts[i], axis=0)
                cj = np.mean(all_verts[j], axis=0)
                d = np.linalg.norm(ci - cj)
                min_dist = 2.0 * np.sqrt(3)
                if d < min_dist:
                    penalty += (min_dist - d) ** 2 * 100.0
    
    # Containment penalty (if any vertex is outside the calculated side)
    # This is technically redundant if we minimize side, but helps guide the optimizer
    # if the side calculation is "too small" due to some numerical issue or if we want to enforce a specific bound.
    # However, since we are minimizing the side that *contains* them, the side is always >= required.
    # The overlap penalty is the main constraint.
    
    return side + penalty

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Try a few outer angles
    outer_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if _remaining_time() < 30.0:
            break
            
        # Multiple starts
        initial_configs = [
            _initial_lattice(),
            _initial_random(),
            _initial_random()
        ]
        
        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break
            
            x0 = []
            for p in init_pos:
                x0.extend(p)
            
            # Nelder-Mead is robust for non-smooth-ish functions if scaled well
            # We use a tight tolerance to squeeze out the last bit of value
            try:
                res = minimize(
                    _objective, 
                    x0, 
                    args=(oa,), 
                    method='Nelder-Mead', 
                    options={
                        'xatol': 1e-6, 
                        'fatol': 1e-6, 
                        'maxiter': 2000, 
                        'maxfev': 2000
                    }
                )
                
                x_opt = res.x
                positions = []
                for i in range(N):
                    positions.append([x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]])
                
                # Verify validity with the computed side
                side = _min_enclosing_side(positions, (0.0, 0.0), oa)
                
                if _check_validity(positions, (0.0, 0.0), side, oa, tol=1e-5):
                    if side < best_score:
                        best_score = side
                        best_result = (positions, (0.0, 0.0), side, oa)
                        
            except Exception as e:
                continue

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
