import numpy as np
import time
import math
from itertools import combinations

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon centered at (0,0)
    given inner hexagons at centers with given angles.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _sat_overlap(p1, p2):
    """
    Check if two convex polygons (lists of (x,y)) overlap using Separating Axis Theorem.
    Returns True if they overlap.
    """
    # We can test axes from both polygons.
    # For efficiency, we can just test the 6 axes of one, but since both are hexagons,
    # their axes are the same set of directions (just rotated).
    # Actually, if angles are different, axes are different.
    # We must test all unique axes.
    # For regular hexagons, the separating axes are the normals to the edges.
    # Edges of hex 1 are perpendicular to directions (ang1 + 30 + k*60).
    # Edges of hex 2 are perpendicular to directions (ang2 + 30 + k*60).
    
    # Let's just collect all normals from both.
    normals = set()
    for ang in [p1[0], p1[3], p2[0], p2[3]]: # p1 is (cx, cy, ang), wait, p1 is vertex list?
        pass # This helper expects vertex lists.
    
    # Let's restructure. We have vertices.
    # Axes are normals to edges.
    # Edge (v[i], v[i+1]) has normal n.
    # We can just generate axes from p1 and p2 edges.
    
    axes = []
    for poly in [p1, p2]:
        for i in range(len(poly)):
            ax = poly[(i+1) % len(poly)][0] - poly[i][0]
            ay = poly[(i+1) % len(poly)][1] - poly[i][1]
            # Normal
            nx = -ay
            ny = ax
            # Normalize to avoid duplicates? Not strictly necessary for SAT, just projection.
            axes.append((nx, ny))
    
    for nx, ny in axes:
        min1 = min(v[0]*nx + v[1]*ny for v in p1)
        max1 = max(v[0]*nx + v[1]*ny for v in p1)
        min2 = min(v[0]*nx + v[1]*ny for v in p2)
        max2 = max(v[0]*nx + v[1]*ny for v in p2)
        
        if max1 + 1e-9 < min2 or max2 + 1e-9 < min1:
            return False
            
    return True

def _check_disjoint(centers, angles):
    """
    Check if all hexagons are disjoint.
    """
    polys = [_hex_vertices(c[0], c[1], a) for c, a in zip(centers, angles)]
    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap(polys[i], polys[j]):
                return False
    return True

def construct():
    start_time = time.time()
    
    # Lattice basis
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Base configuration: Center + 6 neighbors (Ring 1)
    base_centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0.0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    
    # Ring 2 candidates
    ring_2 = [
        (2*np.sqrt(3), 0.0),
        (np.sqrt(3), 3.0),
        (-np.sqrt(3), 3.0),
        (-2*np.sqrt(3), 0.0),
        (-np.sqrt(3), -3.0),
        (np.sqrt(3), -3.0)
    ]
    
    candidates = []
    
    # Option 1: 4 from Ring 2
    for combo in combinations(range(6), 4):
        # Check disjointness among themselves
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                p1 = np.array(ring_2[combo[i]])
                p2 = np.array(ring_2[combo[j]])
                if np.linalg.norm(p1 - p2) < np.sqrt(3) - 1e-9:
                    valid = False
                    break
            if not valid: break
        if valid:
            candidates.append([ring_2[idx] for idx in combo])
            
    best_S = float('inf')
    best_config = None
    
    # Initial coarse search
    for extra_centers in candidates:
        all_centers = base_centers + extra_centers
        all_centers_np = np.array(all_centers)
        
        # Try rotations
        for rot_deg in range(0, 360, 15):
            if time.time() > DEADLINE - 10:
                break
                
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            rotated_centers = all_centers_np @ R.T
            # Center the cluster at origin for the outer hexagon
            shifted_centers = rotated_centers - np.mean(rotated_centers, axis=0)
            
            S = _min_outer_side(list(shifted_centers), [0.0]*N, outer_angle_deg=0.0)
            
            if S < best_S:
                best_S = S
                best_config = (list(shifted_centers), [0.0]*N, 0.0)
                
    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Local Optimization using Nelder-Mead
    # We optimize the positions of the 4 "outer" hexagons (indices 7-10)
    # and potentially the global rotation?
    # The current best_config has a specific rotation baked into the centers.
    # Let's extract the "base" topology and optimize the 4 outer points + global rotation?
    # Actually, it's easier to just optimize the 4 outer points in the current rotated frame.
    # But we might also want to adjust the global rotation.
    # Let's optimize: 8 coords for 4 points + 1 global rotation angle (rad).
    # Total 9 params.
    
    init_centers = np.array(best_config[0])
    init_angles = np.array(best_config[1])
    
    # We need to know which rotation was used to get here to properly parameterize?
    # Or we can just treat the current centers as the starting point and optimize
    # the 4 outer points freely, keeping the inner 7 fixed?
    # If we keep inner 7 fixed, we are only optimizing the outer 4.
    # This is safer and likely sufficient.
    
    def obj(params):
        # params: 8 coords for 4 outer hexagons
        centers = init_centers.copy()
        for k in range(4):
            centers[7+k] = params[2*k:2*k+2]
        
        # Check overlap using SAT
        # This is expensive, so we use a distance check first?
        # No, SAT is robust.
        # But 55 pairs * 12 axes * 6 verts * 2 polys ... ~ 20k ops.
        # 1000 iters * 20k = 20M ops. Should be fine in Python/NumPy mix?
        # Python loops are slow.
        
        # Let's use a faster distance check for early rejection?
        # Min distance for non-overlap is sqrt(3).
        # If dist < sqrt(3) - eps, definitely overlap.
        # If dist > sqrt(3) + eps, might still overlap if rotated?
        # For regular hexagons of side 1, the max distance between centers for overlap
        # is indeed related to the width.
        # Width of hex is 2*sqrt(3).
        # If centers are distance d apart, they overlap if d < width?
        # No, if d < sqrt(3), they definitely overlap.
        # If d > 2*sqrt(3), they definitely don't.
        # In between, it depends on rotation.
        # Since we are fixing angles to 0, the shape is fixed.
        # So we can precompute the Minkowski sum or just use SAT.
        
        # Let's just use SAT. It's the most reliable.
        polys = [_hex_vertices(c[0], c[1], a) for c, a in zip(centers, init_angles)]
        
        for i in range(N):
            for j in range(i+1, N):
                if _sat_overlap(polys[i], polys[j]):
                    return 1e10
        
        S = _min_outer_side(list(centers), init_angles, outer_angle_deg=0.0)
        return S
        
    x0 = init_centers[7:].flatten()
    
    try:
        from scipy.optimize import minimize
        
        result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if result.fun < best_S and result.fun < 1e9:
            new_centers = init_centers.copy()
            for k in range(4):
                new_centers[7+k] = result.x[2*k:2*k+2]
            best_S = result.fun
            best_config = (list(new_centers), [0.0]*N, 0.0)
            
    except Exception:
        pass
        
    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    
    # Ensure we return a valid tuple
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
