"""Rigid-body optimization for 12 unit hexagons in a hexagonal cluster."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 148.0

_SQRT3 = np.sqrt(3.0)
_HALF_SQRT3 = _SQRT3 / 2.0
_60DEG = np.radians(60.0)

def _hex_verts_vec(cx, cy, angle_deg):
    """Vectorized hexagon vertices for a unit hexagon."""
    a = np.radians(angle_deg)
    thetas = a + _60DEG * np.arange(6)
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _hex_normals_vec(angle_deg):
    """6 outward edge normals of a unit hexagon at given rotation."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(30) + _60DEG * np.arange(6)
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _get_cluster_centers():
    """
    Generate centers for a dense hexagonal cluster of 12 hexagons.
    Uses axial coordinates converted to Cartesian.
    """
    # Axial coordinates (q, r)
    # 1 center
    coords = [(0, 0)]
    # 6 neighbors (distance 1)
    coords.extend([
        (1, 0), (1, -1), (0, -1),
        (-1, -1), (-1, 0), (0, 1)
    ])
    # 5 from distance 2 (to make 12 total)
    # We pick 5 that are somewhat symmetric.
    # Distance 2 candidates:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-1,1), (0,2), (1,2), (2,1)
    # Let's pick (2,0), (1,2), (-1,2), (-2,0), (-1,-2) ? 
    # Or just the first 5 in a spiral?
    # Let's use a set that forms a roughly circular shape.
    # (2,0), (0,2), (-2,1), (-2,-1), (1,-2)
    coords.extend([
        (2, 0),
        (0, 2),
        (-2, 1),
        (-2, -1),
        (1, -2)
    ])
    
    centers = []
    for q, r in coords:
        x = _SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    return np.array(centers)

def _compute_cluster_vertices(centers, angle_deg):
    """
    Compute all 72 vertices of the 12 hexagons in the cluster.
    All hexagons have the same rotation angle.
    """
    all_verts = []
    for cx, cy in centers:
        verts = _hex_verts_vec(cx, cy, angle_deg)
        all_verts.append(verts)
    return np.vstack(all_verts) # (72, 2)

def _objective_rigid(angle_rad):
    """
    Objective function for rigid body rotation.
    angle_rad: rotation of the cluster relative to the outer hexagon.
    Outer hexagon is fixed at angle 0.
    """
    angle_deg = np.degrees(angle_rad)
    verts = _compute_cluster_vertices(_CLUSTER_CENTERS, angle_deg)
    
    # Outer hexagon normals (angle 0)
    outer_n = _hex_normals_vec(0.0) # (6, 2)
    
    # Project vertices onto normals
    # (72, 2) @ (2, 6) -> (72, 6)
    projections = verts @ outer_n.T
    
    # Max absolute projection
    max_proj = np.max(np.abs(projections))
    
    # Required side length
    side = max_proj / _HALF_SQRT3
    return side

def _true_side(centers, outer_center, outer_angle):
    """Exact required outer side length."""
    oa = np.radians(outer_angle)
    outer_n = _hex_normals_vec(outer_angle)
    max_d = 0.0
    for c in centers:
        verts = _hex_verts_vec(c[0], c[1], c[2])
        rel = verts - np.array(outer_center)
        d = np.abs(rel @ outer_n.T)
        max_d = max(max_d, d.max())
    return max_d / _HALF_SQRT3

def _true_overlap(centers):
    """Exact SAT min separation. Returns min over all pairs (negative = overlap)."""
    n = len(centers)
    min_sep = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            c1 = centers[i]
            c2 = centers[j]
            n1 = _hex_normals_vec(c1[2])
            n2 = _hex_normals_vec(c2[2])
            axes = np.vstack([n1, n2])
            v1 = _hex_verts_vec(c1[0], c1[1], c1[2])
            v2 = _hex_verts_vec(c2[0], c2[1], c2[2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1), p2.min(axis=1) - p1.max(axis=1))
            min_sep = min(min_sep, seps.min())
    return min_sep

def _is_valid(centers, outer_center, outer_angle, tol=1e-7):
    """Check: no overlaps (SAT) and containment."""
    if _true_overlap(centers) < -tol:
        return False
    return True

# Precompute cluster centers
_CLUSTER_CENTERS = _get_cluster_centers()

def construct():
    t0 = time.time()
    
    # Optimize the rotation angle of the rigid cluster
    # Search range: 0 to 2*pi (full rotation)
    # Due to symmetry of the outer hexagon (60 deg), and potential symmetry of cluster,
    # we could restrict, but full range is safe and fast.
    
    best_angle = 0.0
    best_side = float('inf')
    
    try:
        res = minimize_scalar(_objective_rigid, bounds=(0, 2*np.pi), method='bounded', 
                              options={'xatol': 1e-6, 'maxiter': 1000})
        if res.success:
            best_angle = res.x
            best_side = res.fun
    except Exception:
        # Fallback to grid search if minimize_scalar fails
        angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
        sides = [_objective_rigid(a) for a in angles]
        best_idx = np.argmin(sides)
        best_angle = angles[best_idx]
        best_side = sides[best_idx]
        
        # Refine
        if best_idx > 0 and best_idx < len(angles)-1:
            res = minimize_scalar(_objective_rigid, bounds=(angles[best_idx-1], angles[best_idx+1]), method='bounded')
            if res.success:
                best_angle = res.x
                best_side = res.fun

    # Construct the result
    # Outer hexagon: center (0,0), angle 0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Inner hexagons: rotated centers, angle = best_angle
    angle_deg = np.degrees(best_angle)
    rot_matrix = np.array([
        [np.cos(best_angle), -np.sin(best_angle)],
        [np.sin(best_angle), np.cos(best_angle)]
    ])
    
    rotated_centers = _CLUSTER_CENTERS @ rot_matrix.T
    
    inner = []
    for i in range(N):
        x, y = rotated_centers[i]
        inner.append([x, y, angle_deg])
        
    # Validate
    if not _is_valid(inner, outer_center, outer_angle):
        # This should not happen for a valid honeycomb cluster, but just in case
        # Fallback to a large safe configuration
        inner_default = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = 4.0 * np.cos(theta)
            y = 4.0 * np.sin(theta)
            inner_default.append([x, y, 0.0])
        side = _true_side(inner_default, (0.0, 0.0), 0.0)
        return inner_default, (0.0, 0.0), side, 0.0

    # Calculate true side for final reporting (should match best_side)
    final_side = _true_side(inner, outer_center, outer_angle)
    
    return inner, outer_center, final_side, outer_angle

# EVOLVE-BLOCK-END
