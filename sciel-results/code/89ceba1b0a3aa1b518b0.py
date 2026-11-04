import numpy as np
from scipy.optimize import differential_evolution

def construct():
    N = 12

    # Generate 12 centers from a hexagonal tiling
    # Pointy-top hexagons
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 2
    # Odd columns shifted by 1

    centers = []
    # 3 columns, 4 rows
    for i in range(-1, 2):  # -1, 0, 1
        for j in range(-1, 3):  # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])

    centers = np.array(centers)

    # Center the cluster
    centers -= np.mean(centers, axis=0)

    # Define the 6 vertices of a unit pointy-top hexagon
    # Angles: 90, 150, 210, 270, 330, 30
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Function to calculate required L for a given rotation and center
    def calc_L(rotation_deg, cx, cy):
        rot = np.deg2rad(rotation_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])

        max_L = 0
        for c_center in centers:
            # Rotate center
            rc = R @ c_center
            rc[0] += cx
            rc[1] += cy

            # Rotate vertices
            rv = R @ verts.T
            # Add center
            rv = rv.T + rc

            # Calculate L_req for each vertex
            # L = max(|y|, 2/sqrt(3)*|x|, 2/sqrt(3)*|y + x/sqrt(3)|, 2/sqrt(3)*|y - x/sqrt(3)|)
            x = rv[:, 0]
            y = rv[:, 1]

            L1 = np.abs(y)
            L2 = (2 / np.sqrt(3)) * np.abs(x)
            L3 = (2 / np.sqrt(3)) * np.abs(y + x / np.sqrt(3))
            L4 = (2 / np.sqrt(3)) * np.abs(y - x / np.sqrt(3))

            L_max = np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))
            if L_max > max_L:
                max_L = L_max

        return max_L

    # Use differential evolution for global optimization
    def objective(params):
        rotation_deg, cx, cy = params
        return calc_L(rotation_deg, cx, cy)

    # Bounds for the parameters
    bounds = [(0, 180), (-2, 2), (-2, 2)]

    # Run differential evolution
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-8,
                                    popsize=20, mutation=(0.5, 1.0), recombination=0.7)

    best_rot, best_cx, best_cy = result.x
    best_L = result.fun

    # Construct output
    # Inner hexagons: (x, y, angle)
    # We rotated the centers by best_rot.
    # The hexagons themselves are also rotated by best_rot.
    # But the input format is (x, y, angle_degrees).
    # The angle is the rotation of the hexagon itself.
    # Since we rotated the whole cluster, the hexagon orientation is best_rot.

    rot_rad = np.deg2rad(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])

    inner = []
    for c_center in centers:
        rc = R @ c_center
        rc[0] += best_cx
        rc[1] += best_cy
        inner.append([float(rc[0]), float(rc[1]), float(best_rot)])

    # Outer hexagon: center (best_cx, best_cy), side best_L, rotation 0 (aligned with calc)
    # Note: calc_L assumed container is at 0 rotation (pointy-top).
    # So outer rotation is 0.

    return inner, (float(best_cx), float(best_cy)), float(best_L), 0.0
