import numpy as np
import time

N = 12
SQRT3 = np.sqrt(3)
APOTHEM = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_rad):
        """
        Compute the minimum side length S of the outer hexagon such that all inner hexagons fit.
        S = max over all vertices v and all outer normals n_k of dot(v - c_out, n_k) / APOTHEM
        """
        outer_normals = np.column_stack([
            np.cos(outer_angle_rad + np.radians(np.arange(0, 360, 60))),
            np.sin(outer_angle_rad + np.radians(np.arange(0, 360, 60)))
        ])
        max_proj = 0.0
        for i in range(N):
            verts = hexagon_vertices(inner_centers[i, 0], inner_centers[i, 1], 1.0, inner_angles[i])
            verts_centered = verts - np.array([outer_cx, outer_cy])
            projections = verts_centered @ outer_normals.T
            current_max = np.max(projections)
            if current_max > max_proj:
                max_proj = current_max
        return max_proj / APOTHEM

    def sat_distance(poly1, poly2):
        """
        Separating Axis Theorem distance between two convex polygons.
        Returns min separation (positive if disjoint, negative if overlapping).
        """
        max_gap = -np.inf
        
        def get_normals(poly):
            normals = []
            n = len(poly)
            for i in range(n):
                v1 = poly[i]
                v2 = poly[(i + 1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normals.append(normal / norm)
            return normals
        
        normals1 = get_normals(poly1)
        normals2 = get_normals(poly2)
        
        for normal in normals1:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        for normal in normals2:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def is_valid(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_rad, S, tol=1e-6):
        """Check if all inner hexagons are disjoint and inside the outer hexagon."""
        # Check containment
        actual_S = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_rad)
        if actual_S > S + tol:
            return False, actual_S
        
        # Check disjointness
        verts_list = [hexagon_vertices(inner_centers[i, 0], inner_centers[i, 1], 1.0, inner_angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                dist = sat_distance(verts_list[i], verts_list[j])
                if dist < -tol:
                    return False, actual_S
        return True, actual_S

    # --- Strategy: Kinematic Closure ---
    # We define a family of packings parameterized by:
    # 1. A base configuration (honeycomb-like)
    # 2. A global rotation of the cluster
    # 3. A radial scaling (stretch) factor
    # 4. Possibly individual angle adjustments
    
    # However, the prompt suggests reducing to 2D: global rotation + radial scaling.
    # Let's define a base "honeycomb" cluster of 12 hexagons.
    # A standard honeycomb packing of 12 hexagons:
    # Center hex + 6 surrounding + 5 more? 
    # Actually, 12 hexagons can be packed in a 3x4 grid or other shapes.
    # Let's use a specific dense configuration as the base.
    
    # Base Configuration 1: 3x4 Grid (slightly distorted to fit hexagonal symmetry)
    # In a hexagonal grid, rows are offset.
    # Let's generate a "flower" pattern or a compact cluster.
    
    # Let's try a specific cluster:
    # Center at (0,0).
    # 6 neighbors at distance SQRT3, angles 0, 60, ..., 300.
    # Next shell?
    # Total 7 so far. Need 5 more.
    # This might be sparse.
    
    # Better: Use a known optimal or near-optimal packing for 12 hexagons.
    # A 3x4 rectangular arrangement of hexagons is a common starting point.
    # Let's define the centers for a 3x4 grid of hexagons (side 1).
    # Horizontal spacing: SQRT3
    # Vertical spacing: 1.5
    # Offset for odd rows: SQRT3/2
    
    def generate_grid_config():
        centers = []
        angles = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3
                y = r * 1.5
                if r % 2 == 1:
                    x += SQRT3 / 2
                centers.append([x, y])
                angles.append(0.0)
        centers = np.array(centers)
        # Center the configuration
        centers -= np.mean(centers, axis=0)
        return centers, np.array(angles)

    def generate_circular_config():
        centers = []
        angles = []
        for i in range(12):
            theta = 2 * np.pi * i / 12
            r = 2.5
            centers.append([r * np.cos(theta), r * np.sin(theta)])
            angles.append(theta) # Align hexagons radially?
        centers = np.array(centers)
        centers -= np.mean(centers, axis=0)
        return centers, np.array(angles)

    def transform_config(centers, angles, global_rot, scale, outer_rot):
        """
        Apply global rotation and scaling to the inner cluster.
        The outer hexagon is fixed at origin, rotated by outer_rot.
        We minimize S(centers, angles, outer_rot).
        """
        # Rotate centers
        R = np.array([[np.cos(global_rot), -np.sin(global_rot)],
                      [np.sin(global_rot), np.cos(global_rot)]])
        new_centers = centers @ R.T * scale
        
        # Rotate angles
        new_angles = angles + global_rot
        
        # Compute S
        S = compute_outer_side(new_centers, new_angles, 0.0, 0.0, outer_rot)
        return S, new_centers, new_angles

    best_S = 1e9
    best_centers = None
    best_angles = None
    best_global_rot = 0.0
    best_scale = 1.0
    best_outer_rot = 0.0

    configs = [
        ("grid", *generate_grid_config()),
        ("circular", *generate_circular_config())
    ]

    for name, base_centers, base_angles in configs:
        if time.time() > deadline - 5:
            break
            
        # Search over global_rot, scale, outer_rot
        # Since the outer hexagon has 6-fold symmetry, outer_rot in [0, 30 deg) is sufficient.
        # global_rot in [0, 360 deg) but due to symmetry of cluster, maybe less.
        # Let's do a coarse grid search first.
        
        best_local_S = 1e9
        best_local_params = (0, 1.0, 0)
        
        # Coarse search
        for gr in np.linspace(0, 2*np.pi, 24):
            for sc in np.linspace(0.8, 1.2, 9):
                for orot in np.linspace(0, np.pi/6, 7):
                    S, c, a = transform_config(base_centers, base_angles, gr, sc, orot)
                    # Quick validity check? 
                    # If S is small, likely valid. If S is large, might be invalid due to overlap.
                    # We should check overlap for promising candidates.
                    if S < best_local_S:
                        # Check validity
                        valid, actual_S = is_valid(c, a, 0.0, 0.0, orot, S)
                        if valid:
                            best_local_S = actual_S
                            best_local_params = (gr, sc, orot)
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_global_rot, best_scale, best_outer_rot = best_local_params
            S, best_centers, best_angles = transform_config(base_centers, base_angles, best_global_rot, best_scale, best_outer_rot)
            # Recompute S exactly
            best_S = compute_outer_side(best_centers, best_angles, 0.0, 0.0, best_outer_rot)

        # Fine search around best local
        if best_local_params[0] is not None:
            gr0, sc0, orot0 = best_local_params
            for gr in np.linspace(gr0 - 0.1, gr0 + 0.1, 10):
                for sc in np.linspace(sc0 - 0.05, sc0 + 0.05, 10):
                    for orot in np.linspace(orot0 - 0.05, orot0 + 0.05, 10):
                        S, c, a = transform_config(base_centers, base_angles, gr, sc, orot)
                        valid, actual_S = is_valid(c, a, 0.0, 0.0, orot, S)
                        if valid and actual_S < best_local_S:
                            best_local_S = actual_S
                            best_local_params = (gr, sc, orot)
            
            if best_local_S < best_S:
                best_S = best_local_S
                best_global_rot, best_scale, best_outer_rot = best_local_params
                S, best_centers, best_angles = transform_config(base_centers, base_angles, best_global_rot, best_scale, best_outer_rot)
                best_S = compute_outer_side(best_centers, best_angles, 0.0, 0.0, best_outer_rot)

    # Fallback if no valid configuration found
    if best_centers is None:
        # Use a very loose packing
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    # Final verification
    valid, final_S = is_valid(best_centers, best_angles, 0.0, 0.0, best_outer_rot, best_S)
    if not valid:
        # If the best found is invalid (shouldn't happen if is_valid was strict), use fallback
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    inner = [[best_centers[i, 0], best_centers[i, 1], np.degrees(best_angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), final_S, np.degrees(best_outer_rot)

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
