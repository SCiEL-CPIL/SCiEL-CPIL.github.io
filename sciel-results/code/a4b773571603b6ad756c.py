"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Enumerate hex-lattice subsets, then use smooth optimization
(log-sum-exp soft-max) for the enclosing hexagon, followed by local
perturbation of inner positions and a final high-precision polish.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 148.0  # safety margin before 150s kill


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


def _min_enclosing_side_exact(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons, and return (side, center).
    Uses exact combinatorial solve over active constraint sets.
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
    best_val = float('inf')
    best_c = np.array([0.0, 0.0])
    
    # Try pairs of active constraints
    for i in range(6):
        for j in range(i + 1, 6):
            ni = normals[i]
            nj = normals[j]
            diff_n = ni - nj
            diff_A = A[i] - A[j]
            
            if np.linalg.norm(diff_n) < 1e-12:
                continue
            
            c0 = diff_A / (diff_n @ diff_n) * diff_n
            perp = np.array([-diff_n[1], diff_n[0]])
            perp = perp / np.linalg.norm(perp)
            
            candidates_s = [0.0]
            for k in range(6):
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
                M = np.array([[normals[i][0], normals[i][1], -1],
                              [normals[j][0], normals[j][1], -1],
                              [normals[k][0], normals[k][1], -1]])
                b = np.array([A[i], A[j], A[k]])
                try:
                    sol = np.linalg.solve(M, b)
                    c = sol[:2]
                    t = sol[2]
                    vals = A - c @ normals.T
                    if vals.max() <= t + 1e-9:
                        if t < best_val:
                            best_val = t
                            best_c = c
                except np.linalg.LinAlgError:
                    pass
    
    side = best_val / (sqrt3 / 2)
    return side, best_c


def _min_enclosing_side_smooth(centers, inner_angles, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for the min enclosing side using log-sum-exp.
    Returns (side, center) where side is the soft-max approximation.
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Use the exact solution as starting point
    side_exact, c_exact = _min_enclosing_side_exact(centers, inner_angles, outer_angle_deg)
    
    # Local optimization around c_exact using gradient of soft-max
    c = c_exact.copy()
    lr = 0.01
    for _ in range(50):
        vals = A - c @ normals.T  # shape (6,)
        # soft-max approximation
        m = vals.max()
        log_sum_exp = m + np.log(np.sum(np.exp((vals - m) * beta))) / beta
        t_soft = log_sum_exp / beta
        
        # Gradient: d/dc of soft-max ≈ weighted average of active normals
        weights = np.exp((vals - vals.max()) * beta)
        weights /= weights.sum()
        grad = normals.T @ weights  # gradient of t_soft w.r.t. c (negative)
        
        c -= lr * grad
        lr *= 0.9
    
    vals = A - c @ normals.T
    t = vals.max()
    side = t / (sqrt3 / 2)
    return side, c


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
        side, center = _min_enclosing_side_exact(centers, inner_angles, outer_angle)
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


def _local_optimize(centers, inner_angles, max_iters=100):
    """Perturb inner positions to reduce enclosing side length."""
    best_side, best_center, best_angle = _optimize_outer(centers, inner_angles)
    best_centers = [c[:] for c in centers]
    best_angles = inner_angles[:]
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            for step in step_sizes:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
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
                        if side < best_side - 1e-12:
                            best_side = side
                            best_center = center
                            best_angle = angle
                            best_centers = new_centers
                            improved = True
        
        if not improved:
            break
    
    return best_side, best_center, best_angle, best_centers


def _polish(centers, inner_angles, outer_center, outer_side, outer_angle_deg, max_iters=200):
    """High-precision polish using smaller steps."""
    best_side, best_center, best_angle = _optimize_outer(centers, inner_angles)
    best_centers = [c[:] for c in centers]
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001]
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            for step in step_sizes:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_centers = [c[:] for c in centers]
                        new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                        
                        valid = True
                        for j in range(N):
                            if j != i and _hex_overlap(new_centers[i], inner_angles[i], 
                                                        new_centers[j], inner_angles[j]):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        side, center, angle = _optimize_outer(new_centers, inner_angles)
                        if side < best_side - 1e-14:
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
    
    # Try all C(12,4) = 495 subsets with 0° inner angles
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        side, result = _try_arrangement(centers)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try with 30° rotated inner hexagons
    if time.time() < DEADLINE - 30:
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
    if best_result is not None and time.time() < DEADLINE - 20:
        centers, inner_angles, oc, os_, oa = best_result
        side, center, angle, new_centers = _local_optimize(
            [c[:] for c in centers], inner_angles[:], max_iters=150)
        if side < best_side - 1e-12:
            if _check_validity(new_centers, inner_angles, center, side, angle):
                best_side = side
                best_result = (new_centers, inner_angles, center, side, angle)
    
    # High-precision polish
    if best_result is not None and time.time() < DEADLINE - 5:
        centers, inner_angles, oc, os_, oa = best_result
        side, center, angle, new_centers = _polish(
            [c[:] for c in centers], inner_angles[:], oc, os_, oa, max_iters=200)
        if side < best_side - 1e-14:
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
