"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Define a compact honeycomb arrangement of 11 unit hexagons
    # Centers in a triangular lattice with spacing 2.0
    # Arrangement: a central hexagon surrounded by 6, plus 4 more to make 11
    # Let's use a "flower" pattern: center + 6 neighbors + 4 more
    
    # Honeycomb centers (relative to cluster center)
    # Using axial coordinates converted to Cartesian
    # For hex grid with pointy-top, neighbors are at 60° intervals, distance 2
    # Let's use flat-top orientation for inner hexagons
    
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Center-to-center distance for adjacent hexagons: 2.0
    
    # Define cluster centers in Cartesian coordinates
    # A compact arrangement: 
    # Row structure for 11 hexagons in honeycomb:
    # Let's try: 3-4-3-1 or similar... 
    # Actually, let's use a hexagonal number pattern: 1 + 6 + 4 = 11
    # Or: center, 6 around it, then 4 more
    
    # Better: use a roughly circular/hexagonal cluster
    # Centers:
    centers = []
    # Center
    centers.append((0.0, 0.0))
    # 6 neighbors at distance 2, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = k * np.pi / 3.0
        centers.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    # 4 more: place them at second ring positions
    # Second ring is at distance 2*sqrt(3) ≈ 3.464 from center, but some positions
    # Let's add 4 more at strategic positions
    # Positions at distance 2 from the first ring neighbors, further out
    # Add at angles 30, 90, 150, 210 degrees at distance 2*sqrt(3)
    for k in range(4):
        angle = (2*k + 1) * np.pi / 6.0  # 30, 90, 150, 210
        r = 2.0 * np.sqrt(3)
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    # That's 1 + 6 + 4 = 11 centers
    
    # Re-center the cluster
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx, c[1] - cy) for c in centers]
    
    # All inner hexagons have angle 0 (flat-top)
    inner_angles = [0.0] * 11
    
    # Function to compute vertices of a unit hexagon at (hx, hy) with rotation angle_deg
    def hex_vertices(hx, hy, angle_deg):
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3.0
            vertices.append((hx + np.cos(theta), hy + np.sin(theta)))
        return vertices
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for (hx, hy), ang in zip(centers, inner_angles):
        verts = hex_vertices(hx, hy, ang)
        all_vertices.extend(verts)
    all_vertices = np.array(all_vertices)
    
    # For a regular hexagon with side S, center at origin, rotated by theta,
    # the 6 facet outward normals are at angles theta + k*pi/3 for k=0..5
    # A point p is inside if for all k: dot(p, n_k) <= S * sqrt(3)/2
    # So S = (2/sqrt(3)) * max_k max_p |dot(p, n_k)|
    # We want to minimize S over theta
    
    sqrt3 = np.sqrt(3.0)
    
    def compute_S(theta):
        """Compute required side length for outer hexagon rotated by theta."""
        max_proj = 0.0
        for k in range(6):
            n_angle = theta + k * np.pi / 3.0
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_abs = np.max(np.abs(projections))
            if max_abs > max_proj:
                max_proj = max_abs
        return (2.0 / sqrt3) * max_proj
    
    # Optimize theta: try many values
    best_S = float('inf')
    best_theta = 0.0
    
    # Coarse search
    for i in range(360):
        theta = i * np.pi / 180.0
        S = compute_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Fine search around best
    for i in range(3600):
        theta = best_theta - 0.01 + i * 0.01 * 2.0 / 3600.0
        S = compute_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Even finer
    for i in range(3600):
        theta = best_theta - 0.001 + i * 0.001 * 2.0 / 3600.0
        S = compute_S(theta)
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return with current best
        outer_center = (0.0, 0.0)
        outer_side = best_S
        outer_angle_deg = np.degrees(best_theta)
        inner_list = [[centers[i][0], centers[i][1], inner_angles[i]] for i in range(11)]
        return inner_list, outer_center, outer_side, outer_angle_deg
    
    # Try different cluster arrangements to see if we can do better
    # Alternative: more compact arrangement
    
    # Let's also try a different honeycomb pattern
    # Pattern: 2-3-3-2-1 = 11? No, that's 11
    # Or 1-2-3-3-2 = 11
    
    # Let's try several arrangements and pick the best
    arrangements = []
    
    # Arrangement 1: current one (already computed)
    arrangements.append(('current', best_S, best_theta, centers, inner_angles))
    
    # Arrangement 2: linear-ish but bent
    # Try a more compact hexagonal shape
    # Hexagonal numbers: 1, 7, 19... 11 is between 7 and 19
    # So 11 can't form a perfect hexagon. Best is 7 (center+6) + 4 extra
    
    # Try placing the 4 extras more optimally
    # Alternative: 3-4-3-1 rows
    centers2 = []
    # Row 0 (top): 3 hexagons
    for j in range(3):
        centers2.append(((j - 1) * 2.0, 0.0))
    # Row 1: 4 hexagons, offset by 1 in x, down by sqrt(3)
    for j in range(4):
        centers2.append(((j - 1.5) * 2.0, -np.sqrt(3)))
    # Row 2: 3 hexagons
    for j in range(3):
        centers2.append(((j - 1) * 2.0, -2 * np.sqrt(3)))
    # Row 3: 1 hexagon
    centers2.append((0.0, -3 * np.sqrt(3)))
    
    cx2 = np.mean([c[0] for c in centers2])
    cy2 = np.mean([c[1] for c in centers2])
    centers2 = [(c[0] - cx2, c[1] - cy2) for c in centers2]
    inner_angles2 = [0.0] * 11
    
    all_vertices2 = []
    for (hx, hy), ang in zip(centers2, inner_angles2):
        verts = hex_vertices(hx, hy, ang)
        all_vertices2.extend(verts)
    all_vertices2 = np.array(all_vertices2)
    
    best_S2 = float('inf')
    best_theta2 = 0.0
    for i in range(360):
        theta = i * np.pi / 180.0
        max_proj = 0.0
        for k in range(6):
            n_angle = theta + k * np.pi / 3.0
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            projections = all_vertices2[:, 0] * nx + all_vertices2[:, 1] * ny
            max_abs = np.max(np.abs(projections))
            if max_abs > max_proj:
                max_proj = max_abs
        S = (2.0 / sqrt3) * max_proj
        if S < best_S2:
            best_S2 = S
            best_theta2 = theta
    
    for i in range(3600):
        theta = best_theta2 - 0.01 + i * 0.01 * 2.0 / 3600.0
        max_proj = 0.0
        for k in range(6):
            n_angle = theta + k * np.pi / 3.0
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            projections = all_vertices2[:, 0] * nx + all_vertices2[:, 1] * ny
            max_abs = np.max(np.abs(projections))
            if max_abs > max_proj:
                max_proj = max_abs
        S = (2.0 / sqrt3) * max_proj
        if S < best_S2:
            best_S2 = S
            best_theta2 = theta
    
    arrangements.append(('arr2', best_S2, best_theta2, centers2, inner_angles2))
    
    # Pick the best arrangement
    best_arr = min(arrangements, key=lambda x: x[1])
    _, best_S_final, best_theta_final, centers_final, angles_final = best_arr
    
    outer_center = (0.0, 0.0)
    outer_side = best_S_final
    outer_angle_deg = np.degrees(best_theta_final)
    inner_list = [[centers_final[i][0], centers_final[i][1], angles_final[i]] for i in range(11)]
    
    return inner_list, outer_center, outer_side, outer_angle_deg

# EVOLVE-BLOCK-END
