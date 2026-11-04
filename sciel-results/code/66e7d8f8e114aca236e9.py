"""
Packing 12 unit regular hexagons inside a regular hexagon.
Strategy: Use scipy.optimize.differential_evolution for global search,
with a penalty function for overlaps and containment violations.
Then polish with local optimization (L-BFGS-B) to tighten the fit.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

def hexagon_vertices(center, side, angle_degrees):
    """
    Return vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle_degrees: rotation angle in degrees
    """
    cx, cy = center
    angle = np.deg2rad(angle_degrees)
    # Vertices of a regular hexagon with side length 'side'
    # Flat-top orientation when angle=0: vertices at angles 0, 60, 120, ...
    # Actually, let's use pointy-top or flat-top consistently.
    # Let's define vertices for a hexagon centered at origin with side length 1, flat-top:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This is a flat-top hexagon (horizontal top and bottom edges? No, horizontal left/right points).
    # Let's use the standard: vertices at k*60 degrees.
    angles = np.deg2rad(np.arange(6) * 60 + angle_degrees)
    # For a regular hexagon, distance from center to vertex is equal to side length.
    # So vertices are at distance 'side' from center.
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    return np.column_stack((cx + x, cy + y))

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_polygon(point, vertices):
    """
    Check if a point is inside a convex polygon.
    vertices: (6, 2) array.
    """
    # For convex polygon, check if point is on the same side of all edges.
    # Cross product method.
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        edge = v2 - v1
        vec = point - v1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        # All crosses should have the same sign (or be zero)
        if i == 0:
            sign = np.sign(cross)
            if sign == 0:
                continue
        else:
            if sign * cross < -1e-10:  # Allow small tolerance
                return False
    return True

def polygons_overlap(v1, v2, tol=1e-9):
    """
    Check if two convex polygons overlap.
    Using Separating Axis Theorem (SAT).
    v1, v2: (6, 2) arrays of vertices.
    """
    # Get all edge normals (axes)
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        # Project both polygons onto the axis
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    
    return True

def check_containment(inner_verts, outer_verts):
    """
    Check if all vertices of inner polygon are inside outer polygon.
    """
    for v in inner_verts:
        if not point_in_polygon(v, outer_verts):
            return False
    return True

def evaluate_solution(params, outer_side, outer_angle, outer_center):
    """
    Evaluate a solution. Returns (valid, score) where score is outer_side if valid, else inf.
    params: array of shape (N, 3) -> [x, y, angle_degrees] for each inner hexagon.
    """
    inner_verts_list = []
    for i in range(N):
        x, y, ang = params[i]
        verts = hexagon_vertices((x, y), 1.0, ang)
        inner_verts_list.append(verts)
    
    outer_verts = hexagon_vertices(outer_center, outer_side, outer_angle)
    
    # Check containment
    for verts in inner_verts_list:
        if not check_containment(verts, outer_verts):
            return False, float('inf')
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False, float('inf')
    
    return True, outer_side

def objective(params, outer_side, outer_angle, outer_center):
    """
    Objective function for optimization.
    We fix the outer hexagon at center (0,0) with angle 0 for simplicity?
    Actually, we can optimize outer side, center, and angle too, but let's fix center to (0,0) and angle to 0 to reduce dimensions.
    The problem is symmetric, so this should be fine.
    
    Wait, the task allows rotating and translating the outer hexagon too.
    But due to symmetry, we can fix the outer hexagon's center to (0,0) and its angle to 0 without loss of generality,
    as we can always transform the inner hexagons accordingly.
    
    So we optimize:
    - 12 inner hexagons: 12 * 3 = 36 variables (x, y, angle)
    - 1 outer side length: 1 variable
    
    Total: 37 variables.
    
    However, differential_evolution might struggle with 37 dimensions.
    Let's try a simpler approach first: fix outer side to a guess, and just optimize inner positions.
    Or, use a penalty method.
    
    Let's define the objective as:
    - If valid: return outer_side
    - If invalid: return outer_side + penalty
    
    Penalty = sum of overlap distances + sum of containment violations.
    
    For now, let's just use a hard constraint approach with differential_evolution.
    We'll set bounds for inner hexagon positions to be within a reasonable range.
    
    Bounds:
    - x, y: [-outer_side, outer_side] (roughly)
    - angle: [0, 180] (hexagon has 6-fold symmetry, so 0-60 is enough, but let's use 0-180 to be safe)
    
    Let's start with a guess for outer_side. The area of 12 unit hexagons is 12 * (3*sqrt(3)/2) = 18*sqrt(3) ≈ 31.18.
    Area of outer hexagon with side S is (3*sqrt(3)/2) * S^2.
    So S^2 ≈ 31.18 / (3*sqrt(3)/2) = 31.18 / 2.598 ≈ 12.0.
    So S ≈ 3.46.
    But packing efficiency is less than 100%, so S will be larger. Let's guess S = 4.0 to 6.0.
    """
    # This function is for local optimization polish.
    # For differential_evolution, we need a different objective.
    pass

def de_objective(params):
    """
    Objective for differential_evolution.
    params: [x1, y1, a1, ..., x12, y12, a12, S]
    where S is the outer side length.
    """
    deadline = time.time() + 140  # 140s deadline, 10s margin
    if time.time() > deadline:
        return 1e6
    
    # Extract parameters
    inner_params = params[:36].reshape(12, 3)
    S = params[36]
    
    # Clamp S to a reasonable range
    S = np.clip(S, 3.0, 10.0)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Check validity
    valid, score = evaluate_solution(inner_params, S, outer_angle, outer_center)
    
    if valid:
        return S
    
    # If invalid, return a large penalty
    # We can compute a soft penalty based on how much they overlap or are outside.
    # For now, just return a large number.
    return 1e5 + S

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline
    
    # Define bounds for differential_evolution
    # x, y: [-5, 5] (roughly)
    # angle: [0, 180]
    # S: [3.0, 10.0]
    
    bounds = []
    for i in range(12):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 180.0))  # angle
    bounds.append((3.0, 10.0))  # S
    
    # Run differential_evolution
    result = differential_evolution(
        de_objective,
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-4,
        seed=42,
        polish=False,  # We'll do our own polish
        callback=None
    )
    
    if time.time() > deadline - 20:
        # Not enough time for polish, return current best
        best_params = result.x
        inner_params = best_params[:36].reshape(12, 3)
        S = best_params[36]
        inner = [[float(x), float(y), float(a)] for x, y, a in inner_params]
        return inner, (0.0, 0.0), float(S), 0.0
    
    # Polish with local optimization
    # Use L-BFGS-B to minimize S while keeping validity.
    # This is tricky because the objective is discontinuous.
    # Instead, let's just try a few random perturbations and see if we can reduce S.
    
    best_params = result.x
    best_S = best_params[36]
    
    # Try to reduce S by small amounts and check validity
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        new_S = best_S * (1.0 - 0.01 * np.random.rand())
        if new_S < 3.0:
            continue
        new_params = best_params.copy()
        new_params[36] = new_S
        valid, score = evaluate_solution(new_params[:36].reshape(12, 3), new_S, 0.0, (0.0, 0.0))
        if valid:
            best_S = new_S
            best_params[36] = new_S
    
    inner_params = best_params[:36].reshape(12, 3)
    inner = [[float(x), float(y), float(a)] for x, y, a in inner_params]
    return inner, (0.0, 0.0), float(best_S), 0.0

# For testing
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Inner hexagons: {inner}")
