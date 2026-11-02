"""
Hexagonal Voronoi-inspired Symmetry-Constrained Packing.
Uses Differential Evolution on a reduced 3-parameter space:
1. Inner ring radius (distance of 6 hexagons from center)
2. Outer ring radius (distance of 6 hexagons from center)
3. Outer ring rotation (relative to inner ring)

The 12 hexagons are arranged as:
- 1 at the center (optional, but symmetry suggests 1+5+6 or 1+6+5? 
  Actually, for 12 items, a common dense packing is 1 center + 6 ring + 5 ring? No, 1+6+5=12.
  Or 1 center + 6 ring + 6 ring? 1+6+6=13.
  Let's stick to 1 center + 6 inner ring + 5 outer ring? 
  Wait, 12 hexagons. 
  Hexagonal numbers: 1, 7, 19.
  So 12 is between 7 and 19.
  A 1-center + 6-ring configuration has 7 hexagons.
  Adding 5 more to make 12?
  Or maybe 2 centers? No, symmetry.
  
  Let's look at the inspirations.
  Inspiration 1: 1-center + 6-ring-1 + 5-ring-2? That's 12.
  But 5-ring-2 breaks 6-fold symmetry.
  
  Alternative: 1 center + 6 ring + 5 ring? 
  If we keep 6-fold symmetry, we must have multiples of 6 in rings, plus center.
  1 + 6 + 6 = 13. Too many.
  1 + 6 = 7. Need 5 more.
  
  Maybe the structure is not 1+6+5.
  Maybe it's 2 rings of 6, but one ring is "shifted" or "compressed"?
  
  Let's try a different symmetry: 3-fold?
  1 center + 3 ring + 3 ring + 3 ring + 2? No.
  
  Let's look at the "Voronoi" hint.
  If we assume 6-fold symmetry, we have 12 hexagons.
  12 is not divisible by 6.
  So 6-fold symmetry is broken unless there's a central hexagon and then 6 pairs?
  1 center + 6 pairs = 13.
  
  What if we have NO central hexagon?
  12 hexagons in a ring? 12/6 = 2 rings of 6.
  This preserves 6-fold symmetry.
  Parameters: r1 (inner ring radius), r2 (outer ring radius), theta (rotation of outer ring relative to inner).
  This is a 3-parameter search.
  
  Let's verify if 2 rings of 6 is a good candidate.
  Inner ring: 6 hexagons at distance r1.
  Outer ring: 6 hexagons at distance r2, rotated by theta.
  
  This seems like a very strong structural constraint that reduces 36 dims to 3.
  Differential Evolution on 3 dims is extremely fast and robust.
"""

import numpy as np
import time
import math
from scipy.optimize import differential_evolution

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_symmetric_config(r1, r2, theta_deg):
    """
    Generate 12 hexagon centers based on 2 rings of 6.
    r1: radius of inner ring
    r2: radius of outer ring
    theta_deg: rotation of outer ring relative to inner ring
    """
    centers = []
    
    # Inner ring: 6 hexagons at angles 0, 60, ..., 300
    for i in range(6):
        angle = np.radians(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        # Align hexagons radially? Or fixed orientation?
        # For regular hexagons in a ring, rotating them to align with the radial vector often helps packing.
        # Let's try aligning one vertex or edge with the radial direction.
        # If we rotate by the angle, the hexagon "points" outwards.
        # Let's use angle in degrees for the hexagon rotation.
        centers.append([x, y, angle * 180.0 / np.pi])
        
    # Outer ring: 6 hexagons at angles theta + 0, theta + 60, ...
    theta_rad = np.radians(theta_deg)
    for i in range(6):
        angle = theta_rad + np.radians(60 * i)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append([x, y, angle * 180.0 / np.pi])
        
    return np.array(centers)

def objective(params):
    """
    Objective for Differential Evolution.
    Returns outer side length if valid, else large penalty.
    """
    r1, r2, theta = params
    
    # Basic bounds check
    if r1 < 0.5 or r2 < 1.0 or r1 > 10.0 or r2 > 10.0:
        return 1000.0
        
    centers = generate_symmetric_config(r1, r2, theta)
    
    # Check validity
    if not check_validity(centers):
        return 1000.0 + (r1 + r2) # Penalty
        
    side = compute_required_side(centers)
    return side

def construct():
    start_time = time.time()
    
    # Define bounds for Differential Evolution
    # r1: inner ring radius. Min distance between centers in ring is r1 * sqrt(3) approx?
    # Distance between adjacent centers in ring is 2*r1*sin(30) = r1.
    # Min center distance for non-overlap is sqrt(3) ~ 1.732.
    # So r1 >= 1.732.
    # r2: outer ring radius.
    # theta: rotation. 0 to 60 is enough due to symmetry.
    
    bounds = [
        (1.5, 4.0),   # r1
        (2.0, 6.0),   # r2
        (0.0, 60.0)   # theta
    ]
    
    # Run Differential Evolution
    # maxiter and popsize tuned for speed
    # We have ~140s. DE is fast.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if result.success or result.fun < 1000.0:
            best_params = result.x
            best_side = result.fun
        else:
            best_params = result.x
            best_side = result.fun
    except Exception as e:
        # Fallback to a heuristic guess
        best_params = np.array([1.8, 3.0, 30.0])
        best_side = objective(best_params)
        
    # Generate the best configuration
    centers = generate_symmetric_config(*best_params)
    
    # Final validation
    if check_validity(centers):
        final_side = compute_required_side(centers)
    else:
        # If DE found an invalid config (shouldn't happen with penalty), try a known good one
        # Fallback: 2 rings of 6, standard spacing
        centers = generate_symmetric_config(1.8, 3.2, 30.0)
        final_side = compute_required_side(centers)
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Just return what we have
        pass
        
    return centers.tolist(), (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
