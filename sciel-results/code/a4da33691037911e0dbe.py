# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    deadline = time.time() + 580

    sqrt3 = np.sqrt(3.0)
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, sqrt3])

    first_ring = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    second_ring_all = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]

    best_s = float('inf')
    best_result = None

    for combo in combinations(range(len(second_ring_all)), 5):
        if time.time() > deadline:
            break
        second_ring = [second_ring_all[i] for i in combo]
        lattice_pts = [(0, 0)] + first_ring + second_ring
        centers = np.array([i * a1 + j * a2 for i, j in lattice_pts])

        # Check pairwise distances >= 2 - eps
        valid = True
        for m in range(12):
            for n in range(m + 1, 12):
                d = np.linalg.norm(centers[m] - centers[n])
                if d < 2.0 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Find min enclosing regular hexagon (any angle)
        # For outer hex with center c, angle theta, side s:
        # Vertices: c + s*(cos(theta+60k), sin(theta+60k))
        # Edge normals: theta + 60k + 30 degrees (perpendicular to edges)
        # Actually, for a regular hexagon, the edge normals are at angles theta + 60k + 30? 
        # Let me think: vertex at angle theta, next at theta+60. The edge between them has
        # midpoint at angle theta+30 from center, and the normal points outward at theta+30.
        # So edge normals are at angles theta + 30 + 60k for k=0..5.
        # Inradius (apothem) = s * cos(30 deg) = s * sqrt(3)/2.

        c0 = np.mean(centers, axis=0)
        max_dist = np.max(np.linalg.norm(centers - c0, axis=1))

        best_for_cluster = (float('inf'), None, None)

        # Try several outer angles
        for theta_deg in [0, 15, 30, 45, 60]:
            theta = np.radians(theta_deg)
            normals = np.array([[np.cos(theta + np.radians(30 + 60 * k)),
                                 np.sin(theta + np.radians(30 + 60 * k))] for k in range(6)])

            # Grid search for center
            best_r = float('inf')
            best_c = c0.copy()
            step = max_dist / 8
            for dx in np.linspace(-2.5 * step, 2.5 * step, 15):
                for dy in np.linspace(-2.5 * step, 2.5 * step, 15):
                    c = c0 + np.array([dx, dy])
                    dots = centers @ normals.T
                    shifted = dots - np.dot(c, normals.T)
                    r = np.max(np.abs(shifted))
                    if r < best_r:
                        best_r = r
                        best_c = c.copy()

            # Refine
            try:
                from scipy.optimize import minimize
                def obj(c):
                    dots = centers @ normals.T
                    shifted = dots - np.dot(c, normals.T)
                    return np.max(np.abs(shifted))
                res = minimize(obj, best_c, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_r:
                    best_r = res.fun
                    best_c = res.x
            except Exception:
                pass

            s = best_r / (sqrt3 / 2.0)
            if s < best_for_cluster[0]:
                best_for_cluster = (s, best_c.copy(), theta)

        if best_for_cluster[0] < best_s:
            best_s = best_for_cluster[0]
            best_result = (centers.copy(), best_for_cluster[1], best_for_cluster[2])

    if best_result is None or best_s >= 8:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    centers, outer_c, outer_theta = best_result
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([outer_c[0], outer_c[1], np.degrees(outer_theta)])

    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
