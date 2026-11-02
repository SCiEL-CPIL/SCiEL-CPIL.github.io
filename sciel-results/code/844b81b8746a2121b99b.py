"""Hexagon packing: rigid cluster optimization with smooth support function."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, theta):
    """Get vertices of a regular hexagon with center (cx,cy), circumradius r, rotation theta."""
    angles = theta + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _hex_support(cx, cy, r, theta, dirs):
    """Compute support of hexagon in given directions.
    dirs: array of angles
    Returns array of support values."""
    # Support = max over vertices of dot(vertex, direction_vector)
    # For efficiency, compute using the formula:
    # support(phi) = cx*cos(phi) + cy*sin(phi) + r * max_k cos(phi - theta - k*pi/3)
    cos_phi = np.cos(dirs)
    sin_phi = np.sin(dirs)
    
    base = cx * cos_phi + cy * sin_phi
    
    # For each of the 6 vertex directions, compute cos(phi - (theta + k*pi/3))
    max_inner = np.zeros_like(dirs)
    for k in range(6):
        offset = theta + k * math.pi / 3.0
        vals = r * np.cos(dirs - offset)
        max_inner = np.maximum(max_inner, vals)
    
    return base + max_inner

def _compute_cluster_support(positions, n_dirs=144):
    """Compute max support over all hexagons in n_dirs directions.
    positions: list of (cx, cy, theta)
    Returns array of shape (n_dirs,) with max support in each direction."""
    dirs = np.linspace(0, 2 * math.pi, n_dirs, endpoint=False)
    
    max_support = np.zeros(n_dirs)
    for (cx, cy, th) in positions:
        sup = _hex_support(cx, cy, 1.0, th, dirs)
        max_support = np.maximum(max_support, sup)
    
    return max_support

def _min_enclosing_hex_side(max_support, n_dirs=144):
    """Find minimum enclosing hexagon side length.
    Try all possible outer hexagon rotations (0 to pi/3)."""
    best_side = 1e18
    best_rot = 0.0
    
    # For a regular hexagon with circumradius R and rotation alpha,
    # the 6 side normal directions are at alpha + pi/6 + k*pi/3
    # The apothem (inradius) = R * cos(pi/6) = R * sqrt(3)/2
    # We need: support in each side normal direction <= apothem
    # => R >= support / (sqrt(3)/2)
    
    dir_angles = np.linspace(0, 2 * math.pi, n_dirs, endpoint=False)
    
    # Try rotations from 0 to pi/3 (hexagon has 6-fold symmetry)
    for outer_rot in np.linspace(0, math.pi/3, 36):
        # Side normal directions
        side_normals = outer_rot + math.pi/6.0 + np.arange(6) * math.pi/3.0
        
        max_R = 0.0
        for sn in side_normals:
            # Find nearest direction in our discrete set
            idx = np.argmin(np.abs(dir_angles - sn))
            sup = max_support[idx]
            R_needed = sup / (math.sqrt(3.0) / 2.0)
            if R_needed > max_R:
                max_R = R_needed
        
        # For regular hexagon, side length = circumradius
        if max_R < best_side:
            best_side = max_R
            best_rot = outer_rot
    
    return best_side, best_rot

def _check_no_overlap(positions, min_dist=2.0):
    """Check if any two hexagons overlap.
    For unit hexagons, centers must be at least 2.0 apart for non-overlap
    (this is conservative; actual minimum depends on relative orientation)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist - 1e-9:
                return False
    return True

def _separate_hexagons(positions, min_dist=2.0, max_iters=1000):
    """Simple iterative separation to ensure no overlaps."""
    positions = [list(p) for p in positions]
    n = len(positions)
    
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                
                if d < min_dist:
                    # Push apart
                    if d < 1e-10:
                        # Random direction
                        angle = np.random.uniform(0, 2*math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d
                    
                    push = (min_dist - d) / 2.0 + 0.01
                    positions[i][0] += push * dx
                    positions[i][1] += push * dy
                    positions[j][0] -= push * dx
                    positions[j][1] -= push * dy
                    moved = True
        
        if not moved:
            break
    
    return [tuple(p) for p in positions]

def _objective(params, n_dirs=144):
    """Objective: minimize enclosing hexagon side + large penalty for overlaps."""
    positions = []
    for i in range(N):
        positions.append((params[3*i], params[3*i+1], params[3*i+2]))
    
    # Check overlaps with strong penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < 2.0:
                overlap_penalty += (2.0 - d) ** 2 * 1000.0
    
    if overlap_penalty > 1.0:
        return 1e6 + overlap_penalty
    
    max_support = _compute_cluster_support(positions, n_dirs)
    side, _ = _min_enclosing_hex_side(max_support, n_dirs)
    
    return side + 0.01 * overlap_penalty

def _honeycomb_init(scale=1.0):
    """Generate honeycomb-like initial configuration."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance 2*scale
    for k in range(6):
        angle = k * math.pi / 3.0
        cx = 2.0 * scale * math.cos(angle)
        cy = 2.0 * scale * math.sin(angle)
        positions.append((cx, cy, 0.0))
    
    # Ring 2: 5 more to reach 12 (at distance ~3.46*scale)
    for k in range(5):
        angle = k * 2 * math.pi / 5.0 + math.pi / 5.0
        cx = 3.46 * scale * math.cos(angle)
        cy = 3.46 * scale * math.sin(angle)
        positions.append((cx, cy, 0.0))
    
    return positions

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin for 150s limit
    
    best_side = 1e18
    best_positions = None
    best_outer_rot = 0.0
    
    # Start with honeycomb configuration
    positions = _honeycomb_init(1.0)
    
    # Ensure no overlaps
    positions = _separate_hexagons(positions, min_dist=2.0)
    
    # Evaluate initial
    max_support = _compute_cluster_support(positions, 144)
    side, outer_rot = _min_enclosing_hex_side(max_support, 144)
    
    if side < best_side:
        best_side = side
        best_positions = positions.copy()
        best_outer_rot = outer_rot
    
    # Optimize positions using Nelder-Mead
    def _obj_wrapper(params):
        if time.time() > deadline - 10.0:
            return 1e8
        return _objective(params, 144)
    
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    try:
        result = minimize(_obj_wrapper, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5,
                                 'adaptive': True})
        
        if time.time() < deadline - 5.0:
            # Extract positions
            opt_positions = []
            for i in range(N):
                opt_positions.append((result.x[3*i], result.x[3*i+1], result.x[3*i+2]))
            
            # Ensure no overlaps
            opt_positions = _separate_hexagons(opt_positions, min_dist=2.0)
            
            max_support = _compute_cluster_support(opt_positions, 144)
            side, outer_rot = _min_enclosing_hex_side(max_support, 144)
            
            if _check_no_overlap(opt_positions, 2.0) and side < best_side:
                best_side = side
                best_positions = opt_positions
                best_outer_rot = outer_rot
                
    except Exception:
        pass
    
    # If still no valid solution, use separated honeycomb
    if best_positions is None or not _check_no_overlap(best_positions, 2.0):
        best_positions = _separate_hexagons(_honeycomb_init(1.0), min_dist=2.0)
        max_support = _compute_cluster_support(best_positions, 144)
        best_side, best_outer_rot = _min_enclosing_hex_side(max_support, 144)
    
    # Final verification and output
    if not _check_no_overlap(best_positions, 2.0):
        # Force separation one more time
        best_positions = _separate_hexagons(best_positions, min_dist=2.0, max_iters=2000)
        max_support = _compute_cluster_support(best_positions, 144)
        best_side, best_outer_rot = _min_enclosing_hex_side(max_support, 144)
    
    # Build output
    inner = []
    for (cx, cy, th) in best_positions:
        inner.append([cx, cy, math.degrees(th)])
    
    return inner, (0.0, 0.0), best_side, math.degrees(best_outer_rot)

# EVOLVE-BLOCK-END
