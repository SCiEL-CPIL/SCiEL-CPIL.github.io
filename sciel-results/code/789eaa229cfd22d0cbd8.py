import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Unit hexagon vertices (flat-top, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])
    
    # Outer hexagon edge normals (for a hexagon with side L, the inradius is L*sqrt(3)/2)
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees
    # For aligned inner hexagons, the relevant normals for separation are the same 6 directions
    # plus the 6 directions rotated by 60? No, for two aligned hexagons, the separating axis
    # is one of the edge normals of the hexagon.
    edge_normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    
    # Precompute support values for unit hexagon for each edge normal
    # support[k] = max(dot(unit_verts, normal[k]))
    # min_support[k] = min(dot(unit_verts, normal[k]))
    support_vals = np.array([np.dot(unit_verts, n) for n in edge_normals])
    min_support_vals = np.array([np.dot(unit_verts, n) for n in edge_normals])
    # Actually, for a regular hexagon centered at origin, max proj is R (circumradius) = 1?
    # No, circumradius of unit side hex is 1.
    # Let's verify: vertex at (1,0). Normal at 0 deg? No, normals are at 30, 90...
    # Normal at 90 deg is (0,1). Max y of unit hex is sqrt(3)/2.
    # So support_vals should be sqrt(3)/2 for all normals?
    # Let's check:
    # Normal 30 deg: (sqrt(3)/2, 1/2).
    # Dot with (1,0) = sqrt(3)/2.
    # Dot with (1/2, sqrt(3)/2) = 3/4 + sqrt(3)/4 ~ 1.18? No.
    # The support function of a regular hexagon with side 1 in direction theta is:
    # If theta is aligned with a vertex normal, it's 1.
    # If theta is aligned with an edge normal, it's sqrt(3)/2.
    # Our normals are edge normals (30, 90, etc).
    # So support is constant sqrt(3)/2 for all 6 normals?
    # Let's calculate:
    # n0 = (cos30, sin30) = (0.866, 0.5)
    # v0 = (1, 0) -> dot = 0.866
    # v1 = (0.5, 0.866) -> dot = 0.433 + 0.433 = 0.866
    # v2 = (-0.5, 0.866) -> dot = -0.433 + 0.433 = 0
    # v3 = (-1, 0) -> dot = -0.866
    # v4 = (-0.5, -0.866) -> dot = -0.866
    # v5 = (0.5, -0.866) -> dot = 0.433 - 0.433 = 0
    # Max is 0.866 = sqrt(3)/2.
    # Min is -0.866.
    # So for aligned hexagons, the width in any edge normal direction is sqrt(3).
    
    # Generate hexagonal lattice points (axial coordinates)
    def generate_lattice(R):
        pts = []
        for q in range(-R, R+1):
            for r in range(-R, R+1):
                if abs(q + r) <= R:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    pts.append((x, y))
        return pts
    
    # R=3 gives 19 points. This is the standard "hexagon of hexagons".
    lattice_R3 = generate_lattice(3)
    
    def compute_min_L(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Compute minimum side length L of outer hexagon enclosing all unit hexagons."""
        cx, cy = outer_center
        ca, sa = math.cos(outer_angle), math.sin(outer_angle)
        
        # Rotated edge normals
        normals = np.zeros((6, 2))
        for k in range(6):
            theta = outer_angle + math.radians(30 + k*60)
            normals[k] = [math.cos(theta), math.sin(theta)]
        
        max_proj = 0.0
        for px, py in centers:
            # Relative position to outer center
            rx, ry = px - cx, py - cy
            for k in range(6):
                nx, ny = normals[k]
                # Projection of center
                proj_center = rx * nx + ry * ny
                # Support of unit hexagon in direction n
                # We need max over vertices of dot(R(0)*v + (px,py), n)
                # = max_j dot(unit_verts[j], n) + proj_center
                # Since unit_verts are fixed, we can precompute support for each normal direction
                # But normals change with outer_angle.
                # However, the support of a regular hexagon in direction theta depends only on theta mod 60.
                # For a unit hexagon, support(theta) = cos(theta - nearest_vertex_angle) * ... 
                # Actually, it's easier to just compute it.
                # But 12 hexagons * 6 normals = 72 dot products per L evaluation.
                # In refinement, this is called many times.
                
                # Optimization: 
                # The support of a unit hexagon in direction n is:
                # Let alpha = angle of n.
                # The support is max( cos(alpha), cos(alpha-60), ..., cos(alpha-300) )? No.
                # The vertices are at 0, 60, ...
                # Support = max_j cos(alpha - j*60)
                # This function is periodic and piecewise cosine.
                
                dots = unit_verts @ np.array([nx, ny])
                max_dot = dots.max()
                proj = proj_center + max_dot
                if proj > max_proj:
                    max_proj = proj
        
        # L such that inradius = L*sqrt(3)/2 = max_proj
        # L = 2*max_proj/sqrt(3)
        return 2.0 * max_proj / sqrt3
    
    def hexagons_overlap(p1, p2, eps=1e-10):
        """SAT check for two aligned unit hexagons."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        # For aligned hexagons, check 6 edge normals
        for k in range(6):
            nx, ny = edge_normals[k]
            # Projection of hex 1
            # Center 1 proj: p1 . n
            # Extent: sqrt(3)/2
            proj1_min = p1[0]*nx + p1[1]*ny - sqrt3/2
            proj1_max = p1[0]*nx + p1[1]*ny + sqrt3/2
            # Projection of hex 2
            proj2_min = p2[0]*nx + p2[1]*ny - sqrt3/2
            proj2_max = p2[0]*nx + p2[1]*ny + sqrt3/2
            
            if proj1_max + eps < proj2_min or proj2_max + eps < proj1_min:
                return False
        return True
    
    def all_non_overlapping(centers):
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                if hexagons_overlap(centers[i], centers[j]):
                    return False
        return True
    
    best_L = float('inf')
    best_config = None
    
    # Step 1: Try all 12-subsets of the 19-point lattice (R=3)
    n_pts = len(lattice_R3)
    if n_pts >= 12:
        for subset in combinations(range(n_pts), 12):
            if time.time() > deadline - 20:
                break
            centers = [lattice_R3[i] for i in subset]
            if all_non_overlapping(centers):
                L = compute_min_L(centers)
                if L < best_L:
                    best_L = L
                    best_config = centers[:]
    
    # Step 2: Continuous refinement
    if best_config is not None and time.time() < deadline - 5:
        def objective(params):
            if time.time() > deadline:
                return 1e10
            centers = [(params[i*2], params[i*2+1]) for i in range(12)]
            oc = (params[24], params[25])
            oa = params[26]
            
            L = compute_min_L(centers, oc, oa)
            
            # Penalty for overlap
            # Use a smooth penalty based on distance
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for non-overlap of aligned hexagons is sqrt(3)
                    min_dist = sqrt3
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            return L + 50.0 * penalty
        
        x0 = np.array([c[0] for c in best_config] + [c[1] for c in best_config] + [0.0, 0.0, 0.0])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_L and res.fun < 1e6:
                centers_ref = [(res.x[i*2], res.x[i*2+1]) for i in range(12)]
                L_ref = compute_min_L(centers_ref, (res.x[24], res.x[25]), res.x[26])
                if all_non_overlapping(centers_ref):
                    best_L = L_ref
                    best_config = centers_ref
        except Exception:
            pass
    
    if best_config is None:
        # Fallback: 12 points from R=3 lattice (first 12)
        best_config = [lattice_R3[i] for i in range(12)]
        best_L = compute_min_L(best_config)
    
    return (
        [[best_config[i][0], best_config[i][1], 0.0] for i in range(12)],
        (0.0, 0.0),
        best_L,
        0.0
    )
