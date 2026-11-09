import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _min_distance_between_hexes(pos1, ang1, pos2, ang2):
    """
    Compute the minimum distance between two regular hexagons of side 1.
    Returns the minimum distance between their boundaries (0 if overlapping/touching).
    Uses support function approach: min over all vertex-edge pairs.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], ang1, 1.0)
    v2 = _hex_vertices(pos2[0], pos2[1], ang2, 1.0)
    
    min_dist = np.inf
    # For each edge of hex1, compute distance to all vertices of hex2
    for i in range(6):
        a = v1[i]
        b = v1[(i+1) % 6]
        ab = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-10:
            continue
        ab_unit = ab / ab_len
        
        for j in range(6):
            p = v2[j]
            # Project p onto line ab
            t = np.dot(p - a, ab_unit)
            t = np.clip(t, 0, ab_len)
            closest = a + t * ab_unit
            d = np.linalg.norm(p - closest)
            if d < min_dist:
                min_dist = d
        
        # Also check vertices of hex1 to edges of hex2 (already covered by symmetry if we do all pairs)
    
    # Also check vertex-vertex distances
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_dist:
                min_dist = d
    
    return min_dist

def _min_distance_to_boundary(pos, ang, outer_center, outer_side, outer_angle):
    """
    Compute the minimum distance from a unit hexagon to the boundary of the outer hexagon.
    The outer hexagon has center outer_center, side length outer_side, rotation outer_angle.
    Returns the minimum distance from the inner hexagon's vertices/edges to the outer boundary.
    If the inner hexagon is outside, returns negative.
    """
    v_inner = _hex_vertices(pos[0], pos[1], ang, 1.0)
    
    # Outer hexagon vertices
    v_outer = _hex_vertices(outer_center[0], outer_center[1], outer_angle, outer_side)
    
    # Check if all inner vertices are inside the outer hexagon
    # For a regular hexagon, a point is inside if its projection onto each of the 6 normals
    # is within the apothem distance from the center.
    normals_angles = outer_angle + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    apothem = outer_side * np.sqrt(3.0) / 2.0
    
    min_dist_to_boundary = np.inf
    any_outside = False
    
    for v in v_inner:
        rel = v - outer_center
        proj = np.abs(rel @ normals.T)
        # The maximum projection should be <= apothem for the point to be inside
        max_proj = proj.max()
        if max_proj > apothem + 1e-10:
            any_outside = True
            # Distance to boundary is apothem - max_proj (negative if outside)
            dist = apothem - max_proj
            if dist < min_dist_to_boundary:
                min_dist_to_boundary = dist
        else:
            # Point is inside; distance to boundary is apothem - max_proj
            dist = apothem - max_proj
            if dist < min_dist_to_boundary:
                min_dist_to_boundary = dist
    
    if any_outside:
        return min_dist_to_boundary
    
    # If all inside, also check edges of inner hexagon against outer boundary
    # For simplicity, vertex check is a good approximation for minimum distance
    return min_dist_to_boundary

def _packing_quality(positions, angles, outer_center, outer_side, outer_angle):
    """
    Compute a quality measure: minimum of all pairwise distances between inner hexes
    and distances from inner hexes to outer boundary.
    We want this to be >= 0 for a valid packing.
    """
    min_pair_dist = np.inf
    min_boundary_dist = np.inf
    
    for i in range(N):
        # Distance to boundary
        bd = _min_distance_to_boundary(positions[i], angles[i], outer_center, outer_side, outer_angle)
        if bd < min_boundary_dist:
            min_boundary_dist = bd
        
        for j in range(i+1, N):
            pd = _min_distance_between_hexes(positions[i], angles[i], positions[j], angles[j])
            if pd < min_pair_dist:
                min_pair_dist = pd
    
    return min(min_pair_dist, min_boundary_dist)

def _objective(v):
    """
    v: [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta, side]
    """
    hex_vars = v[:3*N]
    outer_vars = v[3*N:]
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = hex_vars[i*3]
        positions[i, 1] = hex_vars[i*3+1]
        angles[i] = hex_vars[i*3+2]
        
    cx = outer_vars[0]
    cy = outer_vars[1]
    theta = outer_vars[2]
    side = outer_vars[3]
    
    outer_center = np.array([cx, cy])
    
    # We want to minimize the outer side, but with a penalty for invalid packings
    # Penalty: if any distance is negative, add a large penalty
    min_dist = _packing_quality(positions, angles, outer_center, side, theta)
    
    if min_dist < 0:
        penalty = 100.0 * min_dist ** 2
    else:
        penalty = 0.0
    
    return side + penalty

def _generate_initial_guesses():
    """Generate several initial configurations for optimization."""
    guesses = []
    
    # 1. Hexagonal lattice arrangement
    axial_coords = [(0,0)]
    axial_coords += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    axial_coords += [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    
    selected_axial = axial_coords[:N]
    
    lattice_centers = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers.append((x, y))
    
    positions = np.array(lattice_centers)
    angles = np.zeros(N)
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    theta = 0.0
    
    # Estimate side length
    max_r = np.max(np.linalg.norm(positions - np.array([cx, cy]), axis=1))
    side_est = max_r * 2.0 / np.sqrt(3.0) + 1.0
    
    v = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions[:,0], positions[:,1], angles)]).flatten(),
        [cx, cy, theta, side_est]
    ])
    guesses.append(v)
    
    # 2. Circular arrangement
    for r in [2.0, 2.5, 3.0]:
        positions_circ = []
        angles_circ = []
        positions_circ.append((0.0, 0.0))
        angles_circ.append(0.0)
        for k in range(1, N):
            ang = 2.0 * np.pi * (k-1) / (N-1)
            positions_circ.append((r * np.cos(ang), r * np.sin(ang)))
            angles_circ.append(0.0)
        
        positions_circ = np.array(positions_circ)
        angles_circ = np.array(angles_circ)
        cx = np.mean(positions_circ[:, 0])
        cy = np.mean(positions_circ[:, 1])
        max_r = np.max(np.linalg.norm(positions_circ - np.array([cx, cy]), axis=1))
        side_est = max_r * 2.0 / np.sqrt(3.0) + 1.0
        
        v = np.concatenate([
            np.array([[x, y, a] for x, y, a in zip(positions_circ[:,0], positions_circ[:,1], angles_circ)]).flatten(),
            [cx, cy, 0.0, side_est]
        ])
        guesses.append(v)
    
    # 3. Random perturbations of lattice
    for _ in range(10):
        v_perturbed = guesses[0] + np.random.randn(len(guesses[0])) * 0.2
        guesses.append(v_perturbed)
    
    return guesses

def construct():
    start_time = time.time()
    best_result = None
    best_side = float('inf')
    
    guesses = _generate_initial_guesses()
    
    # Use multiple optimization methods
    for v0 in guesses:
        if time.time() > _DEADLINE - 15:
            break
            
        try:
            # Try L-BFGS-B
            res = minimize(_objective, v0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.fun < best_side:
                v_opt = res.x
                hex_vars = v_opt[:3*N]
                outer_vars = v_opt[3*N:]
                
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = hex_vars[i*3]
                    positions[i, 1] = hex_vars[i*3+1]
                    angles[i] = hex_vars[i*3+2]
                    
                cx = outer_vars[0]
                cy = outer_vars[1]
                theta = outer_vars[2]
                side = outer_vars[3]
                
                # Check validity
                min_dist = _packing_quality(positions, angles, np.array([cx, cy]), side, theta)
                
                if min_dist >= -1e-5 and side < best_side:
                    best_side = side
                    inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                    best_result = (inner, (cx, cy), side, np.degrees(theta))
                    
        except Exception as e:
            pass
    
    # If we have time, try differential evolution for a more global search
    if time.time() < _DEADLINE - 30:
        try:
            # Define bounds for differential evolution
            # Positions: within a reasonable range
            # Angles: 0 to 360 degrees
            # Outer center: within range
            # Outer angle: 0 to 360
            # Outer side: 2 to 10
            
            bounds = []
            for i in range(N):
                bounds.extend([(-10, 10), (-10, 10), (0, 2*np.pi)])
            bounds.extend([(-5, 5), (-5, 5), (0, 2*np.pi), (2.0, 10.0)])
            
            def _de_objective(v):
                return _objective(v)
            
            de_res = differential_evolution(_de_objective, bounds, 
                                           maxiter=50, popsize=15, 
                                           tol=1e-6, seed=42, 
                                           polish=True, 
                                           maxfees=1000)
            
            if de_res.fun < best_side:
                v_opt = de_res.x
                hex_vars = v_opt[:3*N]
                outer_vars = v_opt[3*N:]
                
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = hex_vars[i*3]
                    positions[i, 1] = hex_vars[i*3+1]
                    angles[i] = hex_vars[i*3+2]
                    
                cx = outer_vars[0]
                cy = outer_vars[1]
                theta = outer_vars[2]
                side = outer_vars[3]
                
                min_dist = _packing_quality(positions, angles, np.array([cx, cy]), side, theta)
                
                if min_dist >= -1e-5 and side < best_side:
                    best_side = side
                    inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                    best_result = (inner, (cx, cy), side, np.degrees(theta))
                    
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0
        
    return best_result
