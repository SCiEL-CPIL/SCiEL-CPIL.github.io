"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: hexagonal lattice arrangements with continuous optimization of
outer hexagon parameters (center, side, angle) using projection balancing.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 135.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_hex(centers, inner_angles, outer_angle_deg):
    """
    For fixed inner hexagon positions/angles and outer hexagon orientation,
    find the minimum side length and optimal center of the enclosing outer hexagon.
    
    Uses the projection-based formulation:
    - For each of the 6 edge normal directions, compute the max projection of all
      inner vertices onto that direction.
    - The side length constraint is: for each normal n_i, max_p dot(p - c, n_i) <= s*sqrt(3)/2
    - So s >= (A_i - dot(c, n_i)) / (sqrt(3)/2) where A_i = max_p dot(p, n_i)
    - Minimize max_i (A_i - dot(c, n_i)) over c.
    
    This is a convex problem in 2D. We solve it via a bisection + feasibility check,
    or directly via the fact that at optimum, at least 2-3 constraints are tight.
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)  # 6x2
    
    # Collect all vertices of all inner hexagons
    all_points = np.empty((6 * len(centers), 2))
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        all_points[6*i:6*i+6] = verts
    
    # For each normal direction, compute max projection
    projections = all_points @ normals.T  # (6N, 6)
    A = projections.max(axis=0)  # (6,)
    
    # Minimize over c: f(c) = max_i (A_i - c·n_i)
    # This is a 2D convex minimization. Use the fact that the optimum is at
    # a point where at least 2 constraints are tight (vertex of the feasible region).
    # Or use a simple iterative method.
    
    # Better approach: the optimal center c* satisfies that the max is minimized.
    # For a hexagon, the 6 normals come in 3 opposite pairs: n_0=-n_3, n_1=-n_4, n_2=-n_5
    # So A_i - c·n_i and A_{i+3} - c·n_{i+3} = A_{i+3} + c·n_i
    # The constraint becomes: max(A_i - c·n_i, A_{i+3} + c·n_i) <= r for i=0,1,2
    # i.e., |c·n_i - (A_{i+3}-A_i)/2| <= (A_i + A_{i+3})/2 - r ... this gets complex.
    
    # Simplest robust approach: use the subgradient/active set method
    c = np.array([0.0, 0.0])
    
    # Initialize with the centroid of extreme points
    extreme_points = np.empty((6, 2))
    for i in range(6):
        idx = projections[:, i].argmax()
        extreme_points[i] = all_points[idx]
    c = extreme_points.mean(axis=0)
    
    # Iterate: find active constraints, move center
    for _ in range(200):
        vals = A - c @ normals.T  # (6,)
        max_val = vals.max()
        active_mask = vals >= max_val - 1e-12
        n_active = active_mask.sum()
        
        if n_active <= 1:
            break
        
        # At optimum, the weighted sum of active normals should be zero
        # (gradient of max is the average of active normals)
        active_normals = normals[active_mask]
        # The direction to move c is the mean of active normals
        # (since increasing c·n_i for active i reduces their vals)
        grad = active_normals.mean(axis=0)
        
        # Line search: find alpha that minimizes f(c + alpha*grad)
        best_alpha = 0.0
        best_f = max_val
        for alpha in np.linspace(-2.0, 2.0, 41):
            c_try = c + alpha * grad
            f_try = (A - c_try @ normals.T).max()
            if f_try < best_f - 1e-14:
                best_f = f_try
                best_alpha = alpha
        
        if abs(best_alpha) < 1e-10:
            break
        c = c + best_alpha * grad
    
    vals = A - c @ normals.T
    max_proj = vals.max()
    side = max_proj / (sqrt3 / 2.0)
    
    return side, c


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using separating axis theorem."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    # Get edge normals (separating axes) from both hexagons
    axes = []
    for verts in (v1, v2):
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex = x2 - x1
            ey = y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm > 1e-12:
                axes.append((-ey / norm, ex / norm))
    
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """Check if all inner hexagons are inside outer and don't overlap."""
    sqrt3 = np.sqrt(3)
    
    # Check containment
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2.0
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check non-overlap
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True


def _optimize_outer(centers, inner_angles):
    """For fixed inner positions, find optimal outer hexagon (center, side, angle)."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Try a fine range of outer angles (0 to 60 degrees, since hexagon has 6-fold symmetry)
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_hex(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    d1 = np.sqrt(3)  # distance between adjacent hex centers in hex lattice
    d2 = 2 * np.sqrt(3)  # distance for second ring
    
    # First ring: 6 positions
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring: 12 positions
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # Try all C(12,4) = 495 subsets of second ring
    # For each, try inner angles = 0 and inner angles = 30
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        
        for inner_angle_set in ([0.0] * N, [30.0] * N):
            if time.time() > DEADLINE:
                break
            
            # Quick overlap check
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _hex_overlap(centers[i], inner_angle_set[i], centers[j], inner_angle_set[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            side, center, angle = _optimize_outer(centers, inner_angle_set)
            
            if side < best_side:
                # Verify validity
                if _check_validity(centers, inner_angle_set, center, side, angle):
                    best_side = side
                    best_result = (centers, inner_angle_set, center, side, angle)
    
    # Also try slightly perturbed lattice spacings for the best arrangements
    if best_result is not None and time.time() < DEADLINE - 10:
        base_centers, base_angles, _, _, _ = best_result
        # Try small perturbations of the outer angle around the best
        for da in np.linspace(-5, 5, 21):
            if time.time() > DEADLINE:
                break
            # Already handled in _optimize_outer with 121 angle samples
            pass
    
    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
