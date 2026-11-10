# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    deadline = time.time() + 600
    margin = 30

    sqrt3 = np.sqrt(3)
    # Honeycomb lattice vectors for edge-adjacent unit hexagons
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])

    # Unit hexagon vertices (angle 0, centered at origin)
    hex_angles = np.arange(6) * np.pi / 3.0
    unit_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])  # (6,2)

    def make_objective(arr_centers):
        """Create objective function for a given arrangement (12,2) base centers."""
        def objective(params):
            cluster_rot = params[0]
            outer_rot = params[1]

            c, s = np.cos(cluster_rot), np.sin(cluster_rot)
            R = np.array([[c, -s], [s, c]])
            rotated_centers = (R @ arr_centers.T).T  # (12,2)

            # All 72 vertices
            all_verts = (rotated_centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]).reshape(-1, 2)
            centroid = all_verts.mean(axis=0)
            all_verts_c = all_verts - centroid

            def fits(side):
                oa = np.radians(np.degrees(outer_rot) + np.arange(6) * 60.0)
                ov = np.column_stack([side * np.cos(oa), side * np.sin(oa)])
                for i in range(6):
                    p1, p2 = ov[i], ov[(i + 1) % 6]
                    e = p2 - p1
                    cv = e[0] * all_verts_c[:, 1] - e[1] * all_verts_c[:, 0]
                    off = e[1] * p1[0] - e[0] * p1[1]
                    if np.any(cv + off < -1e-10):
                        return False
                return True

            lo, hi = 3.0, 30.0
            if not fits(hi):
                return 1e6
            for _ in range(45):
                mid = (lo + hi) / 2.0
                if fits(mid):
                    hi = mid
                else:
                    lo = mid
            return hi
        return objective

    # Define several arrangements of 12 hexagons in the honeycomb lattice
    arrangements = [
        # A1: center + ring1 + 5 from ring2 (asymmetric)
        [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
         (2,0),(2,-1),(1,-2),(-1,2),(-2,1)],
        # A2: center + ring1 + 5 symmetric from ring2
        [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
         (2,0),(0,-2),(-2,0),(0,2),(1,-2)],
        # A3: 3x4 parallelogram
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),
         (0,2),(1,2),(2,2),(3,0),(3,1),(3,2)],
        # A4: elongated 4x3
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
         (0,-1),(1,-1),(2,-1),(3,-1)],
        # A5: center + ring1 + different 5
        [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
         (2,0),(-2,0),(0,2),(0,-2),(1,2)],
        # A6: 2x6 strip
        [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),
         (0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # A7: center + ring1 + 5 (another choice)
        [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
         (2,0),(2,-1),(-2,1),(-2,0),(0,-2)],
        # A8: diamond-like
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),
         (2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,-1),(1,-1)],
    ]

    np.random.seed(12345)
    overall_best_val = 1e6
    overall_best_params = None
    overall_best_arr = None

    for arr_idx, arr in enumerate(arrangements):
        if time.time() > deadline - margin:
            break

        arr_centers = np.array([q * a1 + r * a2 for q, r in arr])
        obj = make_objective(arr_centers)

        arr_best_val = 1e6
        arr_best_params = None

        n_starts = 20
        for i in range(n_starts):
            if time.time() > deadline - margin:
                break
            if i == 0:
                x0 = np.array([0.0, 0.0])
            elif i == 1:
                x0 = np.array([np.pi / 6, 0.0])
            elif i == 2:
                x0 = np.array([0.0, np.pi / 6])
            elif i == 3:
                x0 = np.array([np.pi / 6, np.pi / 6])
            else:
                x0 = np.array([np.random.uniform(-np.pi, np.pi),
                               np.random.uniform(-np.pi / 3, np.pi / 3)])

            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                if res.fun < arr_best_val:
                    arr_best_val = res.fun
                    arr_best_params = res.x.copy()
            except Exception:
                pass

        if arr_best_val < overall_best_val:
            overall_best_val = arr_best_val
            overall_best_params = arr_best_params
            overall_best_arr = arr

    # Fallback
    if overall_best_params is None or overall_best_val > 100:
        inner_hex_data = np.array(
            [[0,0,0],[-2.5,0,0],[2.5,0,0],[-1.25,2.17,0],[1.25,2.17,0],
             [-1.25,-2.17,0],[1.25,-2.17,0],[-3.75,2.17,0],[3.75,2.17,0],
             [-3.75,-2.17,0],[3.75,-2.17,0],[0,-4,0]]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Reconstruct
    cluster_rot = overall_best_params[0]
    outer_rot = overall_best_params[1]

    c, s = np.cos(cluster_rot), np.sin(cluster_rot)
    R = np.array([[c, -s], [s, c]])

    arr_centers = np.array([q * a1 + r * a2 for q, r in overall_best_arr])
    rotated_centers = (R @ arr_centers.T).T

    all_verts = (rotated_centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]).reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    rotated_centers = rotated_centers - centroid

    inner_angle_deg = np.degrees(cluster_rot)
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [rotated_centers[i, 0], rotated_centers[i, 1], inner_angle_deg]

    outer_hex_data = np.array([0.0, 0.0, np.degrees(outer_rot)])
    outer_hex_side_length = overall_best_val

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
