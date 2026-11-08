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
    
    for i in range(N):
        for j in range(i + 1, N):
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
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None


def _relax_positions(centers, inner_angles, max_iter=200, tol=1e-10):
    """
    Force-directed relaxation to minimize the bounding hexagon side length.
    Returns improved centers and angles.
    """
    centers = [list(c) for c in centers]
    inner_angles = list(inner_angles)
    
    best_side = float('inf')
    best_centers = [list(c) for c in centers]
    best_angles = list(inner_angles)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute current outer hexagon
        side, center, angle = _optimize_outer(centers, inner_angles)
        if side < best_side:
            best_side = side
            best_centers = [list(c) for c in centers]
            best_angles = list(inner_angles)
        
        # Compute gradient of side length w.r.t. each inner hexagon's position
        # and angle by finite differences
        sqrt3 = np.sqrt(3)
        normals = _hex_edge_normals(angle)
        
        # Collect all vertices
        all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                                for i in range(N)])
        projections = all_points @ normals.T
        A = projections.max(axis=0)
        
        # Compute which constraints are active
        vals = A - np.array(center) @ normals.T
        max_val = vals.max()
        active = vals >= max_val - 1e-9
        
        # For each inner hexagon, compute force
        forces = np.zeros((N, 2))
        angle_forces = np.zeros(N)
        
        for i in range(N):
            # Get vertices of this hexagon
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
            
            # For each active constraint, compute contribution to gradient
            for idx in np.where(active)[0]:
                n = normals[idx]
                # Find which vertices are extreme in this direction
                proj = verts @ n
                max_proj = proj.max()
                extreme_idx = np.where(proj >= max_proj - 1e-9)[0]
                
                # The gradient of A_i w.r.t. center is the average of extreme vertices' normals
                # Actually, for a convex polygon, the subgradient of max projection is
                # the convex hull of normals of extreme faces
                # For a hexagon, at most 2 vertices can be extreme in a given direction
                
                # Simplified: push in direction opposite to normal if this hexagon
                # contributes to the max
                if max_proj >= A[idx] - 1e-9:
                    # This hexagon is on the boundary in direction n
                    # Force should push it inward (opposite to n)
                    forces[i] -= n * 0.1
        
        # Also add repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                              (centers[i][1] - centers[j][1])**2)
                min_dist = 2 * np.sqrt(3)  # minimum distance for non-overlap
                if dist < min_dist - 1e-6:
                    # Push apart
                    if dist > 1e-9:
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        force_mag = (min_dist - dist) * 0.5
                        forces[i] += np.array([dx, dy]) / dist * force_mag
                        forces[j] -= np.array([dx, dy]) / dist * force_mag
        
        # Apply forces with decreasing step size
        step = 0.1 * (1 - iteration / max_iter)
        for i in range(N):
            centers[i][0] += forces[i][0] * step
            centers[i][1] += forces[i][1] * step
        
        # Check for overlaps and revert if necessary
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Revert to best
            centers = [list(c) for c in best_centers]
            inner_angles = list(best_angles)
    
    return best_centers, best_angles


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate various hexagonal lattice arrangements
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # First ring
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring positions
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # Try all combinations of 4 from the 12 second-ring positions
    from itertools import combinations as comb
    for subset in comb(range(12), 4):
        if time.time() > DEADLINE:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try with rotated inner hexagons (30 degree rotation)
    if time.time() < DEADLINE - 30:
        for subset in comb(range(12), 4):
            if time.time() > DEADLINE:
                break
            centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
            inner_angles = [30.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Apply force-directed relaxation to the best arrangement
    if best_result is not None and time.time() < DEADLINE - 20:
        centers, inner_angles, outer_center, outer_side, outer_angle = best_result
        relaxed_centers, relaxed_angles = _relax_positions(centers, inner_angles)
        side, result = _try_arrangement(relaxed_centers, relaxed_angles)
        if side < best_side:
            best_side = side
            best_result = result
    
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
