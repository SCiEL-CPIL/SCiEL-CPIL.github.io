"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    n_centers = len(all_centers)  # 19

    # Normal directions for flat-top outer hexagon
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)

    def min_L_for_hexagon(cx, cy, theta):
        """Minimum L for unit hexagon at (cx,cy) with rotation theta in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta minimizing required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse: 0 to 60 in 1-degree steps
        for t_deg in range(0, 61):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine: 3 passes with decreasing step
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            n_steps = int((t_end - t_start) / step) + 1
            for i in range(n_steps):
                t = t_start + i * step
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    SCALE = 1.0000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Precompute: for a given cluster angle, compute optimal theta for each center
    def compute_center_optimals(cluster_angle):
        ca = math.cos(cluster_angle)
        sa = math.sin(cluster_angle)
        results = []
        for i in range(n_centers):
            x, y, _, _ = all_centers[i]
            rx = (x * ca - y * sa) * SCALE
            ry = (x * sa + y * ca) * SCALE
            L_h, theta_h = optimal_theta_for_hexagon(rx, ry)
            results.append((L_h, theta_h))
        return results

    # Phase 1: Search over cluster angles
    # The problem has 60-degree symmetry for the outer hex, but the grid
    # only has 60-degree symmetry itself. We need to search 0 to 60 degrees.
    cluster_angles = []
    # Coarse: every 2 degrees
    for a_deg in range(0, 61, 2):
        cluster_angles.append(math.radians(a_deg))
    # Add finer samples around promising regions
    for a_deg in [1, 3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49, 51, 53, 57, 59]:
        cluster_angles.append(math.radians(a_deg))

    for cluster_angle in cluster_angles:
        if time.time() > deadline - 30.0:
            break

        center_optimal = compute_center_optimals(cluster_angle)

        # Find best subset of 12: we want to minimize max L
        # Sort centers by their L, then the best subset is the 12 with smallest L
        # (since L = max over subset, picking the 12 smallest L values minimizes the max)
        sorted_indices = sorted(range(n_centers), key=lambda i: center_optimal[i][0])
        subset = tuple(sorted_indices[:12])
        L = max(center_optimal[i][0] for i in subset)

        if L < best_L:
            best_L = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
            best_cluster_angle = cluster_angle

    # Phase 2: Joint refinement via coordinate descent
    if time.time() < deadline - 15.0:
        cur_cluster_angle = best_cluster_angle
        cur_thetas = list(best_thetas)
        cur_subset = best_subset
        cur_L = best_L

        for iteration in range(8):
            if time.time() > deadline - 5.0:
                break

            # Step 1: Optimize cluster angle with fixed thetas and subset
            best_ca = cur_cluster_angle
            best_ca_L = cur_L
            # Search around current best angle
            for delta_deg in [-3, -2, -1.5, -1, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1, 1.5, 2, 3]:
                test_ca = cur_cluster_angle + math.radians(delta_deg)
                ca = math.cos(test_ca)
                sa = math.sin(test_ca)
                L_max = 0.0
                for idx, i in enumerate(cur_subset):
                    x, y, _, _ = all_centers[i]
                    rx = (x * ca - y * sa) * SCALE
                    ry = (x * sa + y * ca) * SCALE
                    L_h = min_L_for_hexagon(rx, ry, cur_thetas[idx])
                    if L_h > L_max:
                        L_max = L_h
                if L_max < best_ca_L:
                    best_ca_L = L_max
                    best_ca = test_ca

            if abs(best_ca - cur_cluster_angle) > 1e-9:
                cur_cluster_angle = best_ca
                cur_L = best_ca_L

            # Step 2: Optimize each theta individually
            ca = math.cos(cur_cluster_angle)
            sa = math.sin(cur_cluster_angle)
            for idx, i in enumerate(cur_subset):
                x, y, _, _ = all_centers[i]
                rx = (x * ca - y * sa) * SCALE
                ry = (x * sa + y * ca) * SCALE
                L_h, theta_h = optimal_theta_for_hexagon(rx, ry)
                cur_thetas[idx] = theta_h

            # Recompute L
            ca = math.cos(cur_cluster_angle)
            sa = math.sin(cur_cluster_angle)
            L_new = 0.0
            for idx, i in enumerate(cur_subset):
                x, y, _, _ = all_centers[i]
                rx = (x * ca - y * sa) * SCALE
                ry = (x * sa + y * ca) * SCALE
                L_h = min_L_for_hexagon(rx, ry, cur_thetas[idx])
                if L_h > L_new:
                    L_new = L_h

            if L_new < cur_L - 1e-10:
                cur_L = L_new
            else:
                break

        if cur_L < best_L:
            best_L = cur_L
            best_cluster_angle = cur_cluster_angle
            best_thetas = list(cur_thetas)
            best_subset = cur_subset

    # Phase 3: Try alternative subsets near the boundary
    # The 12 smallest L centers might not be optimal if a slightly larger L center
    # allows better overall packing. Try swapping one center.
    if time.time() < deadline - 10.0:
        ca = math.cos(best_cluster_angle)
        sa = math.sin(best_cluster_angle)
        
        # Compute L for all 19 centers at current cluster angle
        all_L = []
        all_theta = []
        for i in range(n_centers):
            x, y, _, _ = all_centers[i]
            rx = (x * ca - y * sa) * SCALE
            ry = (x * sa + y * ca) * SCALE
            L_h, theta_h = optimal_theta_for_hexagon(rx, ry)
            all_L.append(L_h)
            all_theta.append(theta_h)

        # Try swapping each included center with each excluded center
        included = set(best_subset)
        excluded = [i for i in range(n_centers) if i not in included]
        
        best_swap_L = best_L
        best_swap_subset = best_subset
        best_swap_thetas = list(best_thetas)

        for i_in in best_subset:
            for i_out in excluded:
                if all_L[i_out] >= all_L[i_in]:
                    continue
                # Try this swap
                new_subset = tuple(i if i != i_in else i_out for i in best_subset)
                L_max = max(all_L[i] for i in new_subset)
                if L_max < best_swap_L:
                    best_swap_L = L_max
                    best_swap_subset = new_subset
                    best_swap_thetas = [all_theta[i] for i in new_subset]

        if best_swap_L < best_L:
            best_L = best_swap_L
            best_subset = best_swap_subset
            best_thetas = best_swap_thetas

    # Construct inner hexagons
    inner = []
    ca = math.cos(best_cluster_angle)
    sa = math.sin(best_cluster_angle)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = (x * ca - y * sa) * SCALE
        ry = (x * sa + y * ca) * SCALE
        inner.append([rx, ry, math.degrees(best_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
