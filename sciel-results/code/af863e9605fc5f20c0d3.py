"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Unit hexagon vertices for side length 1, centered at origin, flat-top orientation
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + (side * base_vertices @ R.T)

    def max_projection_to_outer_hex(points, outer_center, outer_angle, outer_side):
        """
        Calculate the minimum side length of an outer hexagon (centered at outer_center, 
        rotated by outer_angle) that contains all 'points'.
        Returns the required side length.
        """
        # Transform points to local frame of outer hexagon (centered at 0,0, angle 0)
        theta = np.deg2rad(outer_angle)
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        
        local_points = (points - outer_center) @ R_inv.T
        
        # The regular hexagon with side S and center (0,0) is defined by:
        # |x| <= S
        # |0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2
        # |-0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2
        
        # Let's compute the required S for each constraint:
        # 1. |x| <= S  =>  S >= max(|x|)
        # 2. |0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2  =>  S >= max(|0.5x + sqrt(3)/2 y|) / (sqrt(3)/2)
        # 3. |-0.5x + sqrt(3)/2 y| <= S * sqrt(3)/2  =>  S >= max(|-0.5x + sqrt(3)/2 y|) / (sqrt(3)/2)
        
        x = local_points[:, 0]
        y = local_points[:, 1]
        
        s1 = np.max(np.abs(x))
        s2 = np.max(np.abs(0.5*x + (np.sqrt(3)/2)*y)) / (np.sqrt(3)/2)
        s3 = np.max(np.abs(-0.5*x + (np.sqrt(3)/2)*y)) / (np.sqrt(3)/2)
        
        return max(s1, s2, s3)

    def evaluate_configuration(positions, angles):
        """
        Given a set of positions and angles for N unit hexagons, 
        return the minimal outer hexagon side length that contains all of them,
        assuming they do not overlap. If they overlap, return a large penalty.
        """
        # First, check for overlaps.
        # For unit hexagons on a triangular lattice with spacing sqrt(3), 
        # they touch at vertices but do not overlap.
        # However, during local search, we might perturb positions.
        # We use a quick bounding box check or vertex containment for robustness.
        
        # To keep it fast, we assume the lattice structure is preserved 
        # and only allow small perturbations that don't cause overlap, 
        # OR we check overlap via distance between centers.
        # Minimum distance between centers of disjoint unit hexagons is sqrt(3) 
        # (if aligned) or 1 (if rotated 30 deg? No, min distance is always > 0).
        # Actually, the minimum distance between centers of two non-overlapping 
        # unit regular hexagons is sqrt(3) if they are aligned (flat-to-flat touching).
        # If rotated, the min distance can be smaller? No, the inradius is sqrt(3)/2.
        # So the distance between centers must be at least 2 * inradius = sqrt(3) 
        # if they are touching flat-to-flat.
        # Wait, if they are rotated relative to each other, they can get closer?
        # No, the width of a hexagon in any direction is between sqrt(3) and 2.
        # The minimum width is sqrt(3) (flat-to-flat).
        # So if two hexagons are disjoint, the distance between their centers 
        # must be at least the sum of their inradii? No, that's for circles.
        # For convex polygons, if they are disjoint, the distance between 
        # any point in P1 and any point in P2 is > 0.
        # A simple sufficient condition for non-overlap is that the distance 
        # between centers is >= 2 * inradius = sqrt(3).
        # This is conservative. It allows for some "wasted space" but ensures validity.
        
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < np.sqrt(3) - 1e-9:
                    return 1e9 # Overlap penalty
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            v = get_hex_verts(positions[i], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Find the minimal outer hexagon containing all vertices
        # We can optimize the outer hexagon's center and angle.
        # Due to symmetry, we can fix the outer center to the centroid of the inner hexagons
        # and try a few angles (0, 30, 60, 90, etc.) or optimize the angle.
        
        # Centroid of inner centers
        outer_center = np.mean(positions, axis=0)
        
        # Try a range of outer angles to find the minimal S
        min_S = 1e9
        # The outer hexagon can be rotated. 
        # Let's try angles from 0 to 180 in steps of 15 degrees, 
        # then refine the best one.
        angles_to_try = np.arange(0, 180, 15)
        best_angle = 0
        for a in angles_to_try:
            S = max_projection_to_outer_hex(all_verts, outer_center, a, 1.0)
            if S < min_S:
                min_S = S
                best_angle = a
        
        # Refine best_angle
        # Use a simple local search on angle
        step = 1.0
        for _ in range(10):
            improved = False
            for da in [-step, step]:
                new_a = best_angle + da
                S = max_projection_to_outer_hex(all_verts, outer_center, new_a, 1.0)
                if S < min_S:
                    min_S = S
                    best_angle = new_a
                    improved = True
            if not improved:
                step /= 2
            if step < 0.1:
                break
                
        return min_S

    def generate_lattice_positions(n, spacing=np.sqrt(3)):
        """
        Generate positions for n hexagons on a triangular lattice.
        We want a compact cluster.
        Triangular lattice vectors:
        v1 = (spacing, 0)
        v2 = (spacing/2, spacing*sqrt(3)/2)
        """
        v1 = np.array([spacing, 0.0])
        v2 = np.array([spacing/2, spacing*np.sqrt(3)/2])
        
        # We'll use a BFS or simply iterate over grid coordinates (i, j)
        # and pick the n closest to the origin.
        # Grid coords (i, j) map to pos = i*v1 + j*v2
        
        candidates = []
        max_range = 10 # Enough for N=11
        for i in range(-max_range, max_range+1):
            for j in range(-max_range, max_range+1):
                pos = i*v1 + j*v2
                dist_to_origin = np.linalg.norm(pos)
                candidates.append((dist_to_origin, i, j, pos))
        
        # Sort by distance to origin
        candidates.sort(key=lambda x: x[0])
        
        # Pick the first N unique positions
        positions = []
        used = set()
        for dist, i, j, pos in candidates:
            key = (i, j)
            if key not in used:
                used.add(key)
                positions.append(pos)
                if len(positions) == n:
                    break
                    
        return np.array(positions)

    # Initial configuration: Triangular lattice
    # Rotate inner hexagons by 30 degrees? 
    # If the lattice is aligned with the hexagon's flat sides, 
    # the flat-to-flat distance is sqrt(3).
    # A standard flat-top hexagon has flat sides horizontal/vertical?
    # No, flat-top means the top and bottom are flat (horizontal).
    # The distance between the top and bottom flat sides is sqrt(3).
    # So if we place them on a triangular lattice with spacing sqrt(3),
    # and align them, they will touch flat-to-flat.
    # Angle 0 for inner hexagons.
    
    positions = generate_lattice_positions(N)
    angles = np.zeros(N)
    
    # Local Search (Hill Climbing)
    # We can perturb the positions slightly to see if it reduces the outer side length.
    # However, perturbing positions might cause overlaps.
    # Given the tight packing, it's unlikely that small perturbations help 
    # unless we break the lattice symmetry, which is risky for validity.
    # A better strategy: Try different "shapes" of the cluster.
    # For N=11, the triangular lattice cluster is likely optimal or near-optimal.
    # Let's just evaluate this configuration.
    
    current_S = evaluate_configuration(positions, angles)
    
    # If the evaluation failed (overlap), try a slightly larger spacing
    if current_S > 1e8:
        # This shouldn't happen with spacing sqrt(3) and angle 0, 
        # but just in case, increase spacing
        positions = generate_lattice_positions(N, spacing=np.sqrt(3) * 1.01)
        current_S = evaluate_configuration(positions, angles)
        
    # Construct the return value
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
        
    # Outer hexagon center is the centroid of the inner positions
    outer_center = np.mean(positions, axis=0)
    
    # We need to return the outer side length and angle.
    # We already found the best angle in evaluate_configuration, but we didn't store it.
    # Let's re-calculate the best angle for the final positions.
    
    all_verts = []
    for i in range(N):
        v = get_hex_verts(positions[i], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    min_S = 1e9
    best_angle = 0
    angles_to_try = np.arange(0, 180, 15)
    for a in angles_to_try:
        S = max_projection_to_outer_hex(all_verts, outer_center, a, 1.0)
        if S < min_S:
            min_S = S
            best_angle = a
            
    # Refine angle
    step = 1.0
    for _ in range(10):
        improved = False
        for da in [-step, step]:
            new_a = best_angle + da
            S = max_projection_to_outer_hex(all_verts, outer_center, new_a, 1.0)
            if S < min_S:
                min_S = S
                best_angle = new_a
                improved = True
        if not improved:
            step /= 2
        if step < 0.1:
            break

    # Final check: ensure time is not exceeded
    if time.time() > deadline:
        # Return a very safe, large hexagon if we ran out of time
        return inner, (0.0, 0.0), 100.0, 0.0

    return inner, (outer_center[0], outer_center[1]), min_S, best_angle

# EVOLVE-BLOCK-END
