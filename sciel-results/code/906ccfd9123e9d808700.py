import numpy as np
import time
import math
import random
from itertools import combinations

N = 11
DEADLINE = 145.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

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

def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons, and return (side, center).
    """
    sqrt3 = np.sqrt(3)
    # Normals of the outer hexagon
    angle = np.radians(outer_angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Minimize max_i (A_i - c·n_i) over c
    # This is a convex problem. The optimal c equalizes the active constraints.
    # Try all pairs and triples of active constraints to find the optimal center c.
    
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

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep outer angle from 0 to 60 degrees
    # Due to symmetry, 0-60 is sufficient.
    # We can use a coarse sweep then refine.
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
            
    return best_side, best_center, best_angle

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = np.empty((6, 2))
    angle = np.radians(outer_angle_deg)
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        outer_normals[i, 0] = np.cos(mid_angle)
        outer_normals[i, 1] = np.sin(mid_angle)
        
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

def _calculate_cost(centers, inner_angles):
    """
    Calculate the cost for a configuration.
    Cost = Outer Side Length + Penalty for Overlaps.
    """
    # Check overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                # Simple penalty: add a large value
                overlap_penalty += 100.0
                # To make it smoother, we could calculate actual overlap area or distance,
                # but SAT boolean is fast. For SA, a constant penalty is often sufficient
                # if the penalty is large enough.
                
    if overlap_penalty > 0:
        # If overlapping, we return a large cost immediately to guide SA away
        # But we still need a side length estimate for continuity?
        # Better: return just the penalty if overlapping, or side + penalty.
        # Let's return side + penalty.
        pass

    side, center, angle = _optimize_outer(centers, inner_angles)
    
    return side + overlap_penalty

def _generate_initial_config():
    """Generate a 'flower' initial configuration."""
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring (6 hexagons)
    d1 = np.sqrt(3) # Distance between centers of touching hexagons
    for i in range(6):
        theta = np.pi / 3 * i
        centers.append([d1 * np.cos(theta), d1 * np.sin(theta)])
        angles.append(0.0)
        
    # Second ring (4 hexagons)
    # The second ring has 12 positions. We pick 4.
    # To maintain symmetry, we can pick 4 that are 90 degrees apart?
    # Or just pick 4 random ones from the 12, or a fixed set.
    # A common tight packing for 11 is 1+6+4.
    # Let's pick 4 positions from the second ring.
    # Second ring distance is 2*sqrt(3).
    d2 = 2 * np.sqrt(3)
    
    # Positions in second ring are at angles:
    # k * 30 deg, with distance d2? No.
    # The lattice vectors are a1 = (sqrt(3), 0) and a2 = (sqrt(3)/2, 1.5).
    # Let's just use the polar coordinates of the lattice points.
    # Ring 2 points:
    # (2*sqrt(3), 0), (sqrt(3), 3), (0, 3*sqrt(3)), etc?
    # Let's generate them properly.
    # Hex lattice points (u, v) -> x = u*sqrt(3) + v*sqrt(3)/2, y = v*1.5
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,0)... wait.
    
    # Let's just use the polar approximation for initialization.
    # 12 positions at distance 2*sqrt(3) ? No, that's not a lattice.
    # The distance to the 2nd ring neighbors is 2*sqrt(3) only for the "corner" ones?
    # Actually, the distance from center to ring 2 centers varies?
    # No, in a hex lattice, all ring 2 centers are at distance 2*sqrt(3) from center?
    # Let's check: (2,0) -> x=2sqrt(3), y=0. Dist = 2sqrt(3).
    # (1,1) -> x=sqrt(3)+sqrt(3)/2 = 1.5sqrt(3), y=1.5. Dist = sqrt(2.7*3 + 2.25) = sqrt(8.1+2.25) = sqrt(10.35) != 2sqrt(3)=sqrt(12).
    # So they are not all at the same distance.
    
    # Let's generate the 12 ring 2 points explicitly.
    ring2_coords = []
    for u in range(-2, 3):
        for v in range(-2, 3):
            # Check if it's in ring 2
            # Hex distance in axial coords (u, v):
            # dist = (|u| + |v| + |u+v|) / 2
            d = (abs(u) + abs(v) + abs(u+v)) / 2
            if d == 2:
                x = u * np.sqrt(3) + v * np.sqrt(3) / 2
                y = v * 1.5
                ring2_coords.append((x, y))
                
    # There should be 12 points.
    # Pick 4. Let's pick 4 that are roughly 90 degrees apart to break symmetry nicely.
    # Or just pick 4 randomly.
    # For deterministic start, let's pick indices 0, 3, 6, 9.
    selected_indices = [0, 3, 6, 9]
    
    for idx in selected_indices:
        x, y = ring2_coords[idx]
        centers.append([x, y])
        angles.append(0.0)
        
    return centers, angles

def construct():
    start_time = time.time()
    
    # Initial Configuration
    centers, inner_angles = _generate_initial_config()
    
    best_side = float('inf')
    best_result = None
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.995
    T_min = 0.001
    
    current_centers = [c[:] for c in centers]
    current_angles = inner_angles[:]
    
    # Initial cost
    current_cost = _calculate_cost(current_centers, current_angles)
    
    if current_cost < best_side:
        # Extract side only (remove penalty if 0)
        side, center, angle = _optimize_outer(current_centers, current_angles)
        if not any(_hex_overlap(current_centers[i], current_angles[i], current_centers[j], current_angles[j]) 
                   for i in range(N) for j in range(i+1, N)):
            best_side = side
            best_result = (current_centers, current_angles, center, side, angle)
    
    while time.time() < DEADLINE:
        # Propose a move
        idx = random.randint(0, N - 1)
        move_type = random.randint(0, 2)
        
        new_centers = [c[:] for c in current_centers]
        new_angles = current_angles[:]
        
        if move_type == 0:
            # Move position
            new_centers[idx][0] += random.uniform(-0.5, 0.5)
            new_centers[idx][1] += random.uniform(-0.5, 0.5)
        elif move_type == 1:
            # Rotate
            new_angles[idx] += random.uniform(-10, 10)
        else:
            # Swap? Or just small nudge
            new_centers[idx][0] += random.uniform(-0.1, 0.1)
            new_centers[idx][1] += random.uniform(-0.1, 0.1)
            new_angles[idx] += random.uniform(-5, 5)
            
        new_cost = _calculate_cost(new_centers, new_angles)
        
        # Acceptance probability
        if new_cost < current_cost:
            current_centers = new_centers
            current_angles = new_angles
            current_cost = new_cost
            
            # Update best if valid (no overlaps)
            # Check validity quickly
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _hex_overlap(current_centers[i], current_angles[i], current_centers[j], current_angles[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                side, center, angle = _optimize_outer(current_centers, current_angles)
                if side < best_side:
                    best_side = side
                    best_result = (current_centers, current_angles, center, side, angle)
        else:
            delta = new_cost - current_cost
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_cost = new_cost
                    
        # Cool down
        T *= cooling_rate
        
    # Fallback if no valid result found
    if best_result is None:
        # Return a linear arrangement
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
