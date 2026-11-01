"""Hexagon packing: place 11 unit regular hexagons inside a regular hexagon, minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import ConvexHull

N = 11

# Unit hexagon vertices (pointy-top, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([[np.cos(np.pi * k / 3), np.sin(np.pi * k / 3)] for k in range(6)])

# Support function of a unit hexagon at rotation theta, in direction u:
# h(theta, u) = max over vertices v of (v + theta) dot u
# For a regular hexagon, the support function in direction u at angle phi is:
# h = max_k (cos(k*pi/3 - phi) + sin(k*pi/3 - phi)*0) ... actually let me think more carefully.

def hex_support(theta, ux, uy):
    """Support function of unit hexagon rotated by theta in direction (ux, uy)."""
    # Rotate vertices by theta
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    verts = _HEX_VERTS @ R.T  # shape (6, 2)
    u = np.array([ux, uy])
    return np.max(verts @ u)

# The six normal directions for a regular hexagon with rotation alpha
def hexagon_normals(alpha):
    """Return the 6 outward normal directions for a regular hexagon rotated by alpha."""
    return np.array([[np.cos(alpha + k * np.pi / 3), np.sin(alpha + k * np.pi / 3)] for k in range(6)])


def min_enclosing_hexagon_side(points, alpha):
    """Compute minimum side length of a regular hexagon with center at origin and rotation alpha
    that encloses all given points."""
    normals = hexagon_normals(alpha)  # shape (6, 2)
    # For each normal direction, compute max support
    supports = np.max(points @ normals.T, axis=1)  # shape (6,)
    # The side length s must satisfy: s * cos(0) >= max_support in that direction
    # Actually, for a regular hexagon with side length s, the support in the normal direction
    # of each face is s (since the distance from center to each face is s*cos(0)=s for a regular hexagon? No.)
    
    # Regular hexagon with side length s: distance from center to each edge = s * sqrt(3)/2
    # The support function in the direction normal to each face is s * sqrt(3)/2
    # Wait, let me reconsider. A regular hexagon with side length s has apothem (center to edge) = s*sqrt(3)/2
    # The support in direction normal to face i is the apothem = s*sqrt(3)/2
    # So we need: s * sqrt(3)/2 >= max_i (max_j point_j . normal_i)
    # Thus s >= 2/sqrt(3) * max_i (max_j point_j . normal_i)
    
    max_support = np.max(supports)
    return 2.0 / np.sqrt(3.0) * max_support


def objective(params, alpha=0.0):
    """Objective: minimize the enclosing hexagon side length for N hexagons."""
    # params: 3*N values: (x, y, theta) for each hexagon
    positions = params.reshape(N, 3)
    
    # Compute all vertices of all hexagons
    all_verts = []
    for i in range(N):
        x, y, theta = positions[i]
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        verts = _HEX_VERTS @ R.T + np.array([x, y])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    # Compute enclosing hexagon side length
    side = min_enclosing_hexagon_side(all_verts, alpha)
    
    # Check pairwise disjointness (minimum distance between any two hexagons)
    # For each pair, compute the minimum distance between their boundaries
    # This is expensive but necessary. Use support function-based check.
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ti = positions[i]
            xj, yj, tj = positions[j]
            
            # Check if hexagons overlap using separating axis theorem
            # The potential separating axes are the 12 edge normals (6 from each hexagon)
            overlap = True
            for theta in [ti, tj]:
                for k in range(6):
                    # Normal direction
                    angle = theta + k * np.pi / 3
                    nx, ny = np.cos(angle), np.sin(angle)
                    
                    # Project both hexagons onto this axis
                    # Hex i
                    ci, si = np.cos(ti), np.sin(ti)
                    Ri = np.array([[ci, -si], [si, ci]])
                    vi = _HEX_VERTS @ Ri.T + np.array([xi, yi])
                    proj_i = vi @ np.array([nx, ny])
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    
                    # Hex j
                    cj, sj = np.cos(tj), np.sin(tj)
                    Rj = np.array([[cj, -sj], [sj, cj]])
                    vj = _HEX_VERTS @ Rj.T + np.array([xj, yj])
                    proj_j = vj @ np.array([nx, ny])
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # Check for separation
                    if max_i < min_j - 1e-9 or max_j < min_i - 1e-9:
                        overlap = False
                        break
                
                if not overlap:
                    break
            
            if overlap:
                # They overlap; add penalty based on how much they overlap
                # Approximate: compute minimum distance between centers minus sum of inradii
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                # Inradius of unit hexagon = sqrt(3)/2
                inradius = np.sqrt(3) / 2
                # If centers are closer than 2*inradius, they likely overlap
                # But this is a rough estimate. Use a smoother penalty.
                # Better: use the actual minimum distance between boundaries
                # For now, use a soft penalty
                penetration = 2 * inradius - dist
                if penetration > 0:
                    penalty += penetration ** 2
    
    return side + 100.0 * penalty


def objective_smooth(params, alpha=0.0, beta=10.0):
    """Smooth surrogate objective."""
    # Compute the enclosing hexagon side length (same as before)
    positions = params.reshape(N, 3)
    
    all_verts = []
    for i in range(N):
        x, y, theta = positions[i]
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        verts = _HEX_VERTS @ R.T + np.array([x, y])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    side = min_enclosing_hexagon_side(all_verts, alpha)
    
    # Smooth penalty for overlaps
    penalty = 0.0
    inradius = np.sqrt(3) / 2
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ti = positions[i]
            xj, yj, tj = positions[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Soft penalty: use log-sum-exp style
            # If dist < 2*inradius, there's likely overlap
            # Use a smooth approximation
            gap = 2 * inradius - dist
            if gap > -1.0:
                penalty += np.exp(beta * gap) / beta
    
    return side + 10.0 * penalty


def initial_guess():
    """Generate an initial guess: hexagons arranged in a honeycomb-like pattern."""
    params = np.zeros(N * 3)
    # Place hexagons in a rough grid
    # For 11 hexagons, try a 3-4-3 or 4-3-4 arrangement
    # Hexagon spacing: adjacent hexagons in honeycomb are 2*inradius = sqrt(3) apart
    spacing = np.sqrt(3) * 1.1  # slightly more to avoid overlap
    
    positions_list = []
    # Row 0: 3 hexagons
    for k in range(3):
        positions_list.append((spacing * k, 0.0, 0.0))
    # Row 1: 4 hexagons, offset
    for k in range(4):
        positions_list.append((spacing * (k - 0.5), spacing * 0.5 * np.sqrt(3), 0.0))
    # Row 2: 3 hexagons
    for k in range(3):
        positions_list.append((spacing * k, spacing * np.sqrt(3), 0.0))
    
    for i, (x, y, theta) in enumerate(positions_list):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = theta
    
    return params


def construct():
    """Construct placement of 11 unit hexagons inside a regular hexagon."""
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 10s margin
    
    # Initial guess
    x0 = initial_guess()
    
    # Try multiple rotations of the outer hexagon
    best_params = None
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_alpha = 0.0
    
    for alpha in [0.0, np.pi / 6, np.pi / 3]:
        # Phase 1: Nelder-Mead optimization with smooth objective
        res = minimize(
            objective_smooth,
            x0,
            args=(alpha,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Phase 2: Polish with exact objective
        res2 = minimize(
            objective,
            res.x,
            args=(alpha,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Use whatever we have
            pass
        
        # Evaluate the result
        positions = res2.x.reshape(N, 3)
        all_verts = []
        for i in range(N):
            x, y, theta = positions[i]
            ct, st = np.cos(theta), np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])
            verts = _HEX_VERTS @ R.T + np.array([x, y])
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        
        # Check for overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ti = positions[i]
                xj, yj, tj = positions[j]
                
                overlap = True
                for theta in [ti, tj]:
                    for k in range(6):
                        angle = theta + k * np.pi / 3
                        nx, ny = np.cos(angle), np.sin(angle)
                        
                        ci, si = np.cos(ti), np.sin(ti)
                        Ri = np.array([[ci, -si], [si, ci]])
                        vi = _HEX_VERTS @ Ri.T + np.array([xi, yi])
                        proj_i = vi @ np.array([nx, ny])
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        
                        cj, sj = np.cos(tj), np.sin(tj)
                        Rj = np.array([[cj, -sj], [sj, cj]])
                        vj = _HEX_VERTS @ Rj.T + np.array([xj, yj])
                        proj_j = vj @ np.array([nx, ny])
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        if max_i < min_j - 1e-9 or max_j < min_i - 1e-9:
                            overlap = False
                            break
                    if not overlap:
                        break
                
                if overlap:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            # Compute the actual enclosing hexagon side
            side = min_enclosing_hexagon_side(all_verts, alpha)
            
            # Center the packing at origin
            centroid = np.mean(all_verts, axis=0)
            
            if side < best_side:
                best_side = side
                best_params = res2.x.copy()
                best_center = tuple(centroid)
                best_alpha = alpha
        
        # Try a few random restarts if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            for restart in range(3):
                x0_r = x0 + 0.5 * np.random.randn(N * 3)
                
                res_r = minimize(
                    objective_smooth,
                    x0_r,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                res_r2 = minimize(
                    objective,
                    res_r.x,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                positions_r = res_r2.x.reshape(N, 3)
                all_verts_r = []
                for i in range(N):
                    x, y, theta = positions_r[i]
                    ct, st = np.cos(theta), np.sin(theta)
                    R = np.array([[ct, -st], [st, ct]])
                    verts = _HEX_VERTS @ R.T + np.array([x, y])
                    all_verts_r.append(verts)
                
                all_verts_r = np.vstack(all_verts_r)
                
                has_overlap_r = False
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ti = positions_r[i]
                        xj, yj, tj = positions_r[j]
                        
                        overlap = True
                        for theta in [ti, tj]:
                            for k in range(6):
                                angle = theta + k * np.pi / 3
                                nx, ny = np.cos(angle), np.sin(angle)
                                
                                ci, si = np.cos(ti), np.sin(ti)
                                Ri = np.array([[ci, -si], [si, ci]])
                                vi = _HEX_VERTS @ Ri.T + np.array([xi, yi])
                                proj_i = vi @ np.array([nx, ny])
                                min_i, max_i = np.min(proj_i), np.max(proj_i)
                                
                                cj, sj = np.cos(tj), np.sin(tj)
                                Rj = np.array([[cj, -sj], [sj, cj]])
                                vj = _HEX_VERTS @ Rj.T + np.array([xj, yj])
                                proj_j = vj @ np.array([nx, ny])
                                min_j, max_j = np.min(proj_j), np.max(proj_j)
                                
                                if max_i < min_j - 1e-9 or max_j < min_i - 1e-9:
                                    overlap = False
                                    break
                            if not overlap:
                                break
                        
                        if overlap:
                            has_overlap_r = True
                            break
                    if has_overlap_r:
                        break
                
                if not has_overlap_r:
                    side_r = min_enclosing_hexagon_side(all_verts_r, alpha)
                    centroid_r = np.mean(all_verts_r, axis=0)
                    
                    if side_r < best_side:
                        best_side = side_r
                        best_params = res_r2.x.copy()
                        best_center = tuple(centroid_r)
                        best_alpha = alpha
    
    # If no valid solution found, fall back to a simple arrangement
    if best_params is None:
        # Use the initial guess and compute its enclosing hexagon
        positions = x0.reshape(N, 3)
        all_verts = []
        for i in range(N):
            x, y, theta = positions[i]
            ct, st = np.cos(theta), np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])
            verts = _HEX_VERTS @ R.T + np.array([x, y])
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        side = min_enclosing_hexagon_side(all_verts, 0.0)
        centroid = np.mean(all_verts, axis=0)
        
        best_params = x0
        best_center = tuple(centroid)
        best_alpha = 0.0
    
    # Extract inner hexagon placements
    positions = best_params.reshape(N, 3)
    inner = []
    for i in range(N):
        x, y, theta = positions[i]
        # Convert theta to degrees
        theta_deg = np.degrees(theta) % 360.0
        inner.append([float(x), float(y), float(theta_deg)])
    
    # The outer hexagon should be centered at the centroid of the packing
    # and have the computed side length
    cx, cy = best_center
    
    return inner, (cx, cy), float(best_side), float(np.degrees(best_alpha)) % 360.0
