import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Pre-defined compact clusters of 12 unit hexagons
    # Coordinates in axial system (q, r) for a hexagonal grid of side length 1
    # Distance between centers of adjacent hexagons is sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Cluster 1: Compact "flower" + 5 more
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 hexagons (partial ring to keep it compact)
    coords1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Cluster 2: Another compact arrangement
    # Try a 3x4 rectangular-ish block in axial coords?
    # Let's generate a few standard compact shapes
    # Shape 2: 2x6 rectangle? No, hex grids don't form rectangles easily.
    # Let's use a "lozenge" or similar.
    # Or just use the same shape but rotated? No, rotation is a global param.
    # Let's define a second distinct compact shape.
    # A 12-hexagon cluster can be a 3x4 rhombus.
    # Rhombus with side lengths 3 and 4?
    # Axial coords for a rhombus of size a x b:
    # q in [0, a-1], r in [0, b-1] ? No, that's not quite right for hex grids.
    # Let's stick to known compact shapes.
    # The most compact shape for 12 is likely the one above (radius 2 hexagon is 1+6+12=19, so 12 is a subset).
    # The subset chosen in coords1 is quite compact.
    
    # Let's add a second variation: a "bar" of 6 and another "bar" of 6?
    # Or a 2x6 strip.
    coords2 = []
    for i in range(6):
        coords2.append((i, 0))
        coords2.append((i, 1))
        
    # We will optimize the global transform for each base cluster.
    base_clusters = [
        np.array([axial_to_cartesian(q, r) for q, r in coords1]),
        np.array([axial_to_cartesian(q, r) for q, r in coords2])
    ]

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon 
        (angle 0, centered at origin) that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def transform_cluster(base_pos, tx, ty, angle_deg):
        """
        Apply global translation and rotation to the base cluster positions.
        """
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x = base_pos[:, 0]
        y = base_pos[:, 1]
        
        new_x = x * cos_a - y * sin_a + tx
        new_y = x * sin_a + y * cos_a + ty
        
        return np.column_stack([new_x, new_y])

    def objective(params):
        """
        Objective function for the rigid body optimization.
        params: [tx, ty, angle_deg]
        Returns the required outer side length.
        """
        tx, ty, angle = params
        pos = transform_cluster(base_pos_global, tx, ty, angle)
        return required_outer_side(pos)

    best_positions = None
    best_side = float('inf')
    best_params = None
    best_base_idx = 0

    # Iterate over base clusters
    for idx, base_pos in enumerate(base_clusters):
        if time.time() > deadline:
            break

        base_pos_global = base_pos.copy()

        # Initial guesses for (tx, ty, angle)
        # Since the outer hexagon is centered at (0,0) and we want to minimize its size,
        # the cluster should be roughly centered at (0,0).
        # The base cluster coords are roughly centered, but let's compute centroid.
        centroid = np.mean(base_pos, axis=0)
        
        # We want the centroid of the transformed cluster to be at (0,0) initially?
        # If we rotate by angle, the centroid moves.
        # Let's just try a few starting points.
        
        initial_guesses = [
            [-centroid[0], -centroid[1], 0],
            [-centroid[0], -centroid[1], 30],
            [-centroid[0], -centroid[1], 60],
            [0, 0, 0],
            [0, 0, 30],
            [0, 0, 60]
        ]

        for guess in initial_guesses:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    objective,
                    guess,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'disp': False
                    }
                )
                
                if res.success or res.fun < best_side:
                    final_side = res.fun
                    if final_side < best_side:
                        best_side = final_side
                        best_params = res.x
                        best_base_idx = idx
            except Exception:
                pass

    if best_positions is None and best_params is not None:
        # Reconstruct best positions
        base_pos = base_clusters[best_base_idx]
        best_positions = transform_cluster(base_pos, best_params[0], best_params[1], best_params[2])
    elif best_positions is None:
        # Fallback: use the first base cluster centered at origin
        base_pos = base_clusters[0]
        centroid = np.mean(base_pos, axis=0)
        best_positions = base_pos - centroid
        best_side = required_outer_side(best_positions)

    # Final validation check (rigid body clusters are disjoint by construction if base is disjoint)
    # But we should verify the outer side calculation is correct.
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
