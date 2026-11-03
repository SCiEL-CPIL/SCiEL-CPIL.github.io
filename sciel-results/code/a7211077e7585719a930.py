"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_rad):
    """Return vertices of regular hexagon with given center, side, rotation."""
    # angle_rad is rotation of the hexagon
    # Vertices are at angles angle_rad + i*pi/3
    angles = angle_rad + np.arange(6) * np.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def min_outer_side_fast(centers, angles_rad, outer_center, outer_angle_rad):
    """
    Compute the minimum side length of an outer regular hexagon 
    (centered at outer_center, rotated by outer_angle_rad) 
    that contains all inner unit hexagons (centers, angles_rad).
    
    The outer hexagon has 6 sides. The distance from center to each side is R = S * sqrt(3)/2.
    For an inner hexagon to be inside, all its vertices must be within distance R from the center 
    in the direction of the 6 outward normals of the outer hexagon.
    
    Actually, it's easier to work with the normals.
    Let the outward normals of the outer hexagon be n_0, ..., n_5.
    n_k = (cos(outer_angle_rad + k*pi/3), sin(outer_angle_rad + k*pi/3))
    
    For an inner hexagon with vertices v_1...v_6, the condition is:
    (v_j - outer_center) . n_k <= R for all j, k.
    
    So for each k, we need R >= max_j ( (v_j - outer_center) . n_k ).
    Let M_k = max over all inner hexagons and their vertices of (v - outer_center) . n_k.
    Then R >= max_k M_k.
    S = 2 * R / sqrt(3).
    """
    # Outer normals
    outer_normals = np.column_stack((
        np.cos(outer_angle_rad + np.arange(6) * np.pi / 3),
        np.sin(outer_angle_rad + np.arange(6) * np.pi / 3)
    ))
    
    # We need to check all vertices of all inner hexagons
    # To be fast, we can pre-compute the vertices of a unit hexagon at angle 0, 
    # then rotate and translate.
    # But angles vary.
    
    max_projection = 0.0
    
    # Precompute cos/sin for inner angles?
    # Let's just loop over inner hexagons. N=12 is small.
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles_rad[i]
        # Vertices of unit hexagon
        # We can compute projections directly
        # v_j = (cx + cos(ang + j*pi/3), cy + sin(ang + j*pi/3))
        # proj = (v_j - outer_center) . n_k
        # = (cx - ocx) * n_kx + (cy - ocy) * n_ky + cos(ang + j*pi/3)*n_kx + sin(ang + j*pi/3)*n_ky
        
        # Relative center
        rx = cx - outer_center[0]
        ry = cy - outer_center[1]
        
        # Base projection of center
        base_proj = np.array([rx * n[0] + ry * n[1] for n in outer_normals])
        
        # Add vertex offsets
        # Vertex offsets for unit hexagon at angle ang
        # Offsets are (cos(ang + j*pi/3), sin(ang + j*pi/3))
        # We need max over j of dot(offset_j, n_k)
        # This is the support function of the unit hexagon in direction n_k, evaluated at rotation ang.
        
        # For a regular hexagon, the support function in direction u is:
        # h(u) = max_j (u . v_j)
        # If we rotate the hexagon by ang, it's h(R_{-ang} u).
        # The support function of a unit hexagon (circumradius 1) in direction phi is:
        # cos(phi mod pi/3 - pi/6) * sqrt(3)/2 ? No.
        # Let's just compute the 6 dot products.
        
        # To be efficient, we can note that the max projection of a unit hexagon rotated by ang 
        # onto a vector n is the same as the max projection of the standard hexagon onto R_{-ang} n.
        # Let's just compute the 6 vertices for each inner hex. 12 * 6 = 72 vertices.
        # 72 * 6 normals = 432 dot products. Very fast.
        
        v_angles = ang + np.arange(6) * np.pi / 3
        vx = np.cos(v_angles)
        vy = np.sin(v_angles)
        
        # Project these 6 vertices onto the 6 outer normals
        # Dot product matrix: (6x6)
        # proj[j, k] = vx[j]*n[k,0] + vy[j]*n[k,1]
        # We need max over j for each k.
        
        # Using numpy broadcasting
        # n is (6, 2), v is (6, 2)
        # proj = v @ n.T -> (6, 6)
        proj = np.column_stack((vx, vy)) @ outer_normals.T
        
        # max_proj_for_directions = max over j (axis 0)
        max_proj = np.max(proj, axis=0)
        
        # Total max projection for direction k is base_proj[k] + max_proj[k]
        total_max = base_proj + max_proj
        
        # Update global max_projection
        current_max = np.max(total_max)
        if current_max > max_projection:
            max_projection = current_max
            
    R = max_projection
    S = 2 * R / np.sqrt(3)
    return S

def overlap_penalty(centers, angles_rad):
    """
    Compute a penalty for overlapping inner hexagons.
    Uses Separating Axis Theorem (SAT) to determine if two convex polygons overlap.
    If they overlap, returns the penetration depth (distance to separate).
    If not, returns 0.
    """
    penalty = 0.0
    n = len(centers)
    
    # Precompute vertices for all hexagons
    verts = []
    for i in range(n):
        verts.append(hex_vertices(centers[i][0], centers[i][1], 1.0, angles_rad[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts[i]
            v2 = verts[j]
            
            # SAT check
            overlap_depth = 0.0
            min_overlap = float('inf')
            
            # Test axes from both polygons
            # Axes are normals to edges
            for poly in [v1, v2]:
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % 6]
                    # Edge vector
                    ex = p2[0] - p1[0]
                    ey = p2[1] - p1[1]
                    # Normal
                    nx = -ey
                    ny = ex
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm < 1e-10:
                        continue
                    nx /= norm
                    ny /= norm
                    
                    # Project v1 and v2 onto axis
                    proj1 = v1 @ np.array([nx, ny])
                    proj2 = v2 @ np.array([nx, ny])
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Check for separation
                    if max1 < min2 or max2 < min1:
                        min_overlap = 0.0
                        break
                    else:
                        # Overlap on this axis
                        depth = min(max1, max2) - max(min1, min2)
                        if depth < min_overlap:
                            min_overlap = depth
                if min_overlap == 0.0:
                    break
            
            if min_overlap > 0.0:
                penalty += min_overlap * min_overlap # Squared penalty for smoothness
        
    return penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    # Side length of unit hex is 1. Distance between centers in hex lattice is sqrt(3).
    d = np.sqrt(3)
    
    # Positions for 12 hexagons:
    # Center (0,0)
    # Ring 1: 6 hexes at distance d
    # Ring 2: 5 hexes at distance 2d (skip one to keep 12 total? 1+6+5=12)
    # Actually, a 3x4 grid or similar might be better for rectangular bounding, 
    # but we have a hex container.
    # Let's use a centered hexagonal cluster: 1 + 6 + 5 = 12.
    
    init_centers = []
    init_angles = []
    
    # Center
    init_centers.append((0.0, 0.0))
    init_angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        init_centers.append((d * np.cos(a), d * np.sin(a)))
        init_angles.append(0.0)
        
    # Ring 2 (partial)
    # We need 5 more.
    # In a full hex lattice, ring 2 has 12 hexes.
    # Let's pick 5 that are roughly symmetric.
    # Indices in ring 2 correspond to angles 0, 30, 60, 90... ?
    # Standard hex lattice ring 2 positions:
    # (2d, 0), (d, d*sqrt(3)), (-d, d*sqrt(3)), (-2d, 0), (-d, -d*sqrt(3)), (d, -d*sqrt(3)) -- these are distance 2d?
    # No, distance from center to (2d, 0) is 2d.
    # Distance to (d, d*sqrt(3)) is sqrt(d^2 + 3d^2) = 2d.
    # So there are 12 positions at distance 2d?
    # Let's list them:
    # 0 deg: (2d, 0)
    # 30 deg: (d*sqrt(3), d) -> dist sqrt(3d^2 + d^2) = 2d
    # 60 deg: (0, 2d)
    # 90 deg: (-d*sqrt(3), d)
    # 120 deg: (-2d, 0)
    # 150 deg: (-d*sqrt(3), -d)
    # 180 deg: (0, -2d)
    # 210 deg: (d*sqrt(3), -d)
    # 240 deg: (2d, 0) ... wait, 12 positions.
    
    # Let's just use the first 5 of these 12 positions, or randomize.
    # To be symmetric, maybe pick every other one?
    # Let's just pick 5 consecutive ones to start, or random.
    # For a better start, let's pick 5 that form a semi-circle or something.
    # Actually, let's just use random restarts with a good initial structure.
    
    # Let's define a function to generate a random perturbed hex lattice
    def generate_initial_config(perturb=0.1):
        centers = []
        angles = []
        
        # Center
        centers.append((0.0, 0.0))
        angles.append(0.0)
        
        # Ring 1
        for i in range(6):
            a = i * np.pi / 3
            x = d * np.cos(a) + random.gauss(0, perturb)
            y = d * np.sin(a) + random.gauss(0, perturb)
            centers.append((x, y))
            angles.append(random.gauss(0, perturb * 10)) # small rotation
            
        # Ring 2: Pick 5 from the 12 available spots
        # Spots are at angles k * pi/6, distance 2d
        # Let's pick 5 random distinct indices from 0..11
        indices = random.sample(range(12), 5)
        for idx in indices:
            a = idx * np.pi / 6
            x = 2 * d * np.cos(a) + random.gauss(0, perturb)
            y = 2 * d * np.sin(a) + random.gauss(0, perturb)
            centers.append((x, y))
            angles.append(random.gauss(0, perturb * 10))
            
        return centers, angles

    def objective(params):
        # params: x1, y1, a1, ..., x12, y12, a12, cx, cy, a_out
        # 12 * 3 + 3 = 39 variables
        
        centers = []
        angles_rad = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3 + 1]
            a = params[i*3 + 2]
            centers.append((x, y))
            angles_rad.append(a)
            
        outer_center = (params[36], params[37])
        outer_angle = params[38]
        
        # Check time
        if time.time() - start_time > DEADLINE:
            return 1000.0
            
        # Compute min side length
        S = min_outer_side_fast(centers, angles_rad, outer_center, outer_angle)
        
        # Compute overlap penalty
        pen = overlap_penalty(centers, angles_rad)
        
        # Objective: S + large_weight * penalty
        # We want S to be small, but penalty must be 0.
        # If penalty > 0, it dominates.
        weight = 10.0
        return S + weight * pen

    best_score = float('inf')
    best_params = None
    
    # Run multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        centers, angles = generate_initial_config(perturb=0.2)
        
        # Initial outer params
        # Center should be near centroid of inner hexes
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        a_out = 0.0
        
        x0 = []
        for c, a in zip(centers, angles):
            x0.append(c[0])
            x0.append(c[1])
            x0.append(a)
        x0.extend([cx, cy, a_out])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
                # Early stop if penalty is 0 and S is very small?
                # Check if valid
                centers_opt = [(best_params[i*3], best_params[i*3+1]) for i in range(12)]
                angles_opt = [best_params[i*3+2] for i in range(12)]
                oc = (best_params[36], best_params[37])
                oa = best_params[38]
                
                S_opt = min_outer_side_fast(centers_opt, angles_opt, oc, oa)
                pen_opt = overlap_penalty(centers_opt, angles_opt)
                
                if pen_opt < 1e-6:
                    # Valid packing
                    # We can break if we find a very good one
                    if S_opt < 4.0: # Arbitrary threshold
                        break
        except Exception:
            continue

    if best_params is None:
        # Fallback
        return [[0,0,0]]*12, (0,0), 10.0, 0.0

    # Extract best result
    centers = [(best_params[i*3], best_params[i*3+1]) for i in range(12)]
    angles_deg = [np.degrees(best_params[i*3+2]) for i in range(12)]
    outer_center = (best_params[36], best_params[37])
    outer_angle_deg = np.degrees(best_params[38])
    
    # Final precise calculation of S
    angles_rad_final = [np.radians(a) for a in angles_deg]
    S_final = min_outer_side_fast(centers, angles_rad_final, outer_center, np.radians(outer_angle_deg))
    
    # Format output
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles_deg)]
    
    return inner, outer_center, S_final, outer_angle_deg

