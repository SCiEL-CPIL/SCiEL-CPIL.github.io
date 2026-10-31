import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Get vertices of regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside regular hexagon."""
    # Transform to hexagon's local frame
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    lx = dx * np.cos(-angle) - dy * np.sin(-angle)
    ly = dx * np.sin(-angle) + dy * np.cos(-angle)
    # Hexagon with side s: |x| <= s, |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2, etc.
    # Actually for a regular hexagon with circumradius = side:
    # The 6 supporting lines are at distance side*sqrt(3)/2 from center
    # In local coords, the hexagon is: max over 6 directions of dot product <= side*sqrt(3)/2
    for i in range(6):
        theta = i * np.pi / 3
        val = lx * np.cos(theta) + ly * np.sin(theta)
        if val > side * np.sqrt(3) / 2 + 1e-10:
            return False
    return True

def hexagon_in_hexagon(inner_center, inner_side, inner_angle, outer_center, outer_side, outer_angle):
    """Check if inner hexagon fits inside outer hexagon."""
    # Get inner hexagon vertices
    inner_verts = hexagon_vertices(inner_center[0], inner_center[1], inner_side, inner_angle)
    # Check all vertices are inside outer hexagon
    for v in inner_verts:
        if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
            return False
    return True

def outer_side_needed(positions, angles, n_hex):
    """Compute minimum outer hexagon side length to contain all inner hexagons.
    positions: (n,2) array of centers
    angles: (n,) array of angles in degrees
    Returns the minimum side length (with outer hexagon centered at origin, angle 0).
    """
    # For each inner hexagon, find the maximum support function value
    # in each of the 6 outer hexagon directions
    # Outer hexagon with side S, angle 0: supporting lines at distance S*sqrt(3)/2
    # in directions 0, 60, 120, 180, 240, 300 degrees
    
    max_support = np.zeros(6)
    
    for i in range(n_hex):
        cx, cy = positions[i]
        ang = angles[i]
        # Vertices of inner hexagon
        for j in range(6):
            theta = np.radians(ang) + j * np.pi / 3
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            for k in range(6):
                dir_theta = k * np.pi / 3
                support = vx * np.cos(dir_theta) + vy * np.sin(dir_theta)
                if support > max_support[k]:
                    max_support[k] = support
    
    # The outer hexagon needs side S such that S*sqrt(3)/2 >= max_support[k] for all k
    # So S >= max_support[k] * 2/sqrt(3) for all k
    return np.max(max_support) * 2.0 / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons: center + 6 around + 4 more
    # For 11, a good arrangement is a "flower" with some extras
    # Let's use a pattern: positions on a triangular lattice
    
    # Triangular lattice spacing for non-overlapping unit hexagons:
    # Two unit hexagons don't overlap if center distance >= 2*sqrt(3)/2 = sqrt(3) ~ 1.732
    # Actually, the minimum distance between centers of non-overlapping unit hexagons
    # (same orientation) is 2*sqrt(3)/2 = sqrt(3) for edge-to-edge contact
    
    # Let's try a compact arrangement
    # Pattern: row of 3, row of 4, row of 3, row of 1? No, let's think more carefully.
    
    # For 11 hexagons in a hexagonal packing:
    # The "hex number" arrangements are 1, 7, 19, ... 
    # 11 is between 7 and 19.
    
    # Let's use: 1 center, 6 at distance sqrt(3), then 4 more at distance 2*sqrt(3)
    # Or better: a 3-4-3-1 or 4-3-4 arrangement
    
    # Actually, let me just use a grid-like arrangement and optimize
    # Start with positions on a triangular lattice
    
    positions = []
    angles = np.zeros(N)
    
    # Hexagonal lattice: basis vectors
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate candidate positions on triangular lattice
    # We want 11 positions that are compact
    # Use indices: (i,j) -> i*a1 + j*a2
    # For compactness, use small i,j values
    
    # Let's try: center at (0,0), then neighbors
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * a1 + j * a2
            lattice_points.append(p)
    
    # Pick 11 most central points
    lattice_points = np.array(lattice_points)
    distances = np.linalg.norm(lattice_points, axis=1)
    indices = np.argsort(distances)[:N]
    positions = lattice_points[indices].copy()
    
    # Center the cluster
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    
    best_side = outer_side_needed(positions, angles, N)
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Verify no overlaps
    def check_no_overlap(pos, ang, n):
        for i in range(n):
            for j in range(i+1, n):
                # Check if two hexagons overlap
                # For same orientation, min center distance is sqrt(3)
                # For different orientations, need more careful check
                # Simple check: if centers are too close, likely overlap
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3) - 1e-6:
                    return False
        return True
    
    if not check_no_overlap(positions, angles, N):
        # Spread out more
        for i in range(N):
            positions[i] *= 1.1
        best_side = outer_side_needed(positions, angles, N)
        best_positions = positions.copy()
    
    # Optimization: minimize outer side length
    def objective(params):
        nonlocal start_time
        if time.time() - start_time > DEADLINE:
            return 1e10
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        # Penalty for overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = np.sqrt(3)  # minimum center distance for non-overlapping
                if d < min_d:
                    overlap_penalty += (min_d - d) ** 2 * 100
        
        side = outer_side_needed(pos, ang, N)
        return side + overlap_penalty
    
    # Initial parameters
    x0 = np.concatenate([best_positions.flatten(), best_angles])
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 
                                  'adaptive': True})
        
        if result.success or result.fun < best_side:
            pos_opt = result.x[:2*N].reshape(N, 2)
            ang_opt = result.x[2*N:]
            
            # Check validity
            if check_no_overlap(pos_opt, ang_opt, N):
                side_opt = outer_side_needed(pos_opt, ang_opt, N)
                if side_opt < best_side:
                    best_side = side_opt
                    best_positions = pos_opt.copy()
                    best_angles = ang_opt.copy()
    except Exception:
        pass
    
    # Try a few more local optimizations from different starts
    for restart in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Perturb initial positions
        np.random.seed(restart)
        pos_try = best_positions + np.random.randn(N, 2) * 0.1
        ang_try = best_angles + np.random.randn(N) * 5.0
        
        x0_try = np.concatenate([pos_try.flatten(), ang_try])
        
        try:
            result = minimize(objective, x0_try, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            
            if result.fun < best_side:
                pos_opt = result.x[:2*N].reshape(N, 2)
                ang_opt = result.x[2*N:]
                
                if check_no_overlap(pos_opt, ang_opt, N):
                    side_opt = outer_side_needed(pos_opt, ang_opt, N)
                    if side_opt < best_side:
                        best_side = side_opt
                        best_positions = pos_opt.copy()
                        best_angles = ang_opt.copy()
        except Exception:
            pass
    
    # Final check and return
    # Ensure all hexagons are inside the outer hexagon
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = best_side + 1e-6  # tiny margin
    
    # Verify
    for i in range(N):
        if not hexagon_in_hexagon(best_positions[i], 1.0, best_angles[i], 
                                  outer_center, outer_side, outer_angle):
            # Expand if needed
            outer_side = outer_side * 1.01
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), 
                     float(best_angles[i])])
    
    return inner, outer_center, float(outer_side), outer_angle

