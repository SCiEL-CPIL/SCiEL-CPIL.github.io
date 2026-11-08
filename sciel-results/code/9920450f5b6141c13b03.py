"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Use hexagonal lattice arrangements, then optimize outer hexagon
via exact projection computation. Also try local perturbation of inner
positions to find tighter packings.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 145.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons, and return (side, center).
    
    For a regular hexagon with side s, inradius r = s*sqrt(3)/2.
    A point p is inside iff dot(p - c, n_i) <= r for all 6 normals n_i.
    So s = max_i (max_p dot(p, n_i) - dot(c, n_i)) / (sqrt(3)/2).
    We minimize over c: min_c max_i (A_i - c·n_i) where A_i = max_p dot(p, n_i).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Minimize max_i (A_i - c·n_i) over c
    # This is a convex problem. The optimal c equalizes the active constraints.
    # For a hexagon, the 6 normals are at 60° apart. At optimum, typically
    # 2-3 constraints are tight.
    
    # Use the fact that at optimum, the subgradient contains 0:
    # sum of active normals weighted = 0
    # Try all pairs and triples of active constraints
    
    best_val = float('inf')
    best_c = np.array([0.0, 0.0])
    
    # Try pairs of active constraints
    for i in range(6):
        for j in range(i + 1, 6):
            # At optimum: A_i - c·n_i = A_j - c·n_j = t
            # c·(n_i - n_j) = A_i - A_j
            # Need another constraint to determine c uniquely in 2D
            # Use the third constraint k that gives the smallest t
            ni = normals[i]
            nj = normals[j]
            diff_n = ni - nj
            diff_A = A[i] - A[j]
            
            # Check if diff_n is non-degenerate
            if np.linalg.norm(diff_n) < 1e-12:
                continue
            
            # Parametrize: c = c0 + s * perp(diff_n)
            # where c0 is any solution to c·diff_n = diff_A
            # Simplest: c0 = diff_A / |diff_n|^2 * diff_n
            c0 = diff_A / (diff_n @ diff_n) * diff_n
            perp = np.array([-diff_n[1], diff_n[0]])
            perp = perp / np.linalg.norm(perp)
            
            # For each s, compute t(s) = max_k (A_k - (c0 + s*perp)·n_k)
            # This is piecewise linear in s. Find minimum.
            # Evaluate at breakpoints where two constraints cross.
            candidates_s = [0.0]
            for k in range(6):
                # A_k - (c0 + s*perp)·n_k = A_k - c0·n_k - s*(perp·n_k)
                # Set equal to constraint i: A_i - c0·n_i - s*(perp·n_i)
                # => s * (perp·n_k - perp·n_i) = (A_k - c0·n_k) - (A_i - c0·n_i)
                denom = perp @ (normals[k] - ni)
                if abs(denom) > 1e-12:
                    s_val = ((A[k] - c0 @ normals[k]) - (A[i] - c0 @ ni)) / denom
                    candidates_s.append(s_val)
            
            for s_val in candidates_s:
                c = c0 + s_val * perp
                vals = A - c @ normals.T
                t = vals.max()
                if t < best_val:
                    best_val = t
                    best_c = c
    
    # Also try triples
    for i in range(6):
        for j in range(i + 1, 6):
            for k in range(j + 1, 6):
                # Solve c·n_i = A_i - t, c·n_j = A_j - t, c·n_k = A_k - t
                # This is 3 equations, 3 unknowns (c_x, c_y, t)
                M = np.array([[normals[i][0], normals[i][1], -1],
                              [normals[j][0], normals[j][1], -1],
                              [normals[k][0], normals[k][1], -1]])
                b = np.array([A[i], A[j], A[k]])
                try:
                    sol = np.linalg.solve(M, b)
                    c = sol[:2]
                    t = sol[2]
                    # Verify this is indeed the max
                    vals = A - c @ normals.T
                    if vals.max() <= t + 1e-9:
                        if t < best_val:
                            best_val = t
                            best_c = c
                except np.linalg.LinAlgError:
                    pass
    
    side = best_val / (sqrt3 / 2)
    return side, best_c


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True


def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None


def _local_optimize(centers, inner_angles, max_iters=200):
    """Perturb inner positions to reduce enclosing side length."""
    best_side, best_center, best_angle = _optimize_outer(centers, inner_angles)
    best_centers = [c[:] for c in centers]
    best_angles = inner_angles[:]
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            for dx in [-0.05, 0.05]:
                for dy in [-0.05, 0.05]:
                    new_centers = [c[:] for c in centers]
                    new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                    
                    # Check overlaps
                    valid = True
                    for j in range(N):
                        if j != i and _hex_overlap(new_centers[i], inner_angles[i], 
                                                    new_centers[j], inner_angles[j]):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    side, center, angle = _optimize_outer(new_centers, inner_angles)
                    if side < best_side - 1e-10:
                        best_side = side
                        best_center = center
                        best_angle = angle
                        best_centers = new_centers
                        improved = True
        
        if not improved:
            break
    
    return best_side, best_center, best_angle, best_centers


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # First ring positions
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring positions (12 total)
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # Try all C(12,4) = 495 subsets
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        side, result = _try_arrangement(centers)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try with 30° rotated inner hexagons
    if time.time() < DEADLINE - 20:
        for subset in combinations(range(12), 4):
            if time.time() > DEADLINE:
                break
            centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
            inner_angles = [30.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Local optimization on best result
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, inner_angles, oc, os_, oa = best_result
        side, center, angle, new_centers = _local_optimize(
            [c[:] for c in centers], inner_angles[:], max_iters=100)
        if side < best_side - 1e-10:
            if _check_validity(new_centers, inner_angles, center, side, angle):
                best_side = side
                best_result = (new_centers, inner_angles, center, side, angle)
    
    # Fallback
    if best_result is None:
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
