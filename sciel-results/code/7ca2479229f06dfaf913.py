"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster (flat-top hexagons)
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        # Vertex angles relative to hexagon center
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])

        # Projection of each vertex onto each normal direction
        # proj[i,j] = vx[i]*cos(normal_j) + vy[i]*sin(normal_j)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period is 60 degrees due to hexagonal symmetry)
        # Coarse search
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Fine search around best with multiple passes
        for step in [1.0, 0.1, 0.01, 0.001]:
            t_min = max(0.0, best_theta - step * 5)
            t_max = min(math.radians(60), best_theta + step * 5)
            for t_deg in np.arange(t_min, t_max + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap
    SCALE = 1.0000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But the optimal might be at 30 degrees (rotating the "corners" to align with "sides")
    cluster_angles_to_try = []
    # Coarse search
    for a_deg in range(0, 61, 5):
        cluster_angles_to_try.append(math.radians(a_deg))
    # Add key angles explicitly
    for a_deg in [15, 30, 45, 52.5]:
        cluster_angles_to_try.append(math.radians(a_deg))
    # Deduplicate
    cluster_angles_to_try = sorted(set(cluster_angles_to_try))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline:
            break

        # Rotate all centers by cluster_angle
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * math.cos(cluster_angle) - y * math.sin(cluster_angle)
            ry = x * math.sin(cluster_angle) + y * math.cos(cluster_angle)
            rotated_centers.append((rx, ry, q, r))

        # Precompute optimal theta for each center (independent of subset)
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        
        # Sort centers by their individual L requirement to help subset selection
        sorted_indices = sorted(range(19), key=lambda i: center_optimal[i][0])

        # Try all subsets of size 12 from 19 centers
        # Use sorted order to prune: if the 12th smallest L is already >= best_L, skip
        for subset in combinations(range(19), 12):
            if time.time() > deadline:
                break
            # L is the max over all hexagons in subset
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle
                # Early termination if we find a very good solution
                if best_L < 9.9:
                    break

    if best_subset is None:
        # Fallback: use first 12 with no cluster rotation
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Build initial positions from best discrete solution
    init_positions = np.zeros((N, 2))
    init_thetas = np.zeros(N)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = x * math.cos(best_cluster_angle) - y * math.sin(best_cluster_angle)
        ry = x * math.sin(best_cluster_angle) + y * math.cos(best_cluster_angle)
        init_positions[idx, 0] = rx * SCALE
        init_positions[idx, 1] = ry * SCALE
        init_thetas[idx] = best_thetas[idx]

    # Smooth optimization using scipy if available
    if HAS_SCIPY and time.time() < deadline - 10.0:
        def min_separation(dtheta):
            """
            Compute the minimum center-to-center distance for two unit hexagons
            with relative rotation dtheta to be non-overlapping.
            """
            dtheta = abs(dtheta) % (2 * math.pi)
            dtheta = min(dtheta, 2*math.pi - dtheta)
            
            # For two regular hexagons of side 1:
            # The support function in direction phi is max over vertices of dot(v, u_phi)
            # where u_phi = (cos(phi), sin(phi))
            # For hexagon 1 with rotation theta1, vertices at angles theta1 + k*pi/3
            # For hexagon 2 with rotation theta2, vertices at angles theta2 + k*pi/3
            # 
            # The minimum separation is the minimum over all directions phi of:
            #   h1(phi) + h2(phi + pi)
            # where h1 is the support function of hexagon 1 in direction phi
            # and h2 is the support function of hexagon 2 in direction phi+pi
            #
            # For a unit regular hexagon with rotation theta:
            # h(phi) = max_{k=0..5} cos(phi - theta - k*pi/3)
            #        = max over 6 values
            #
            # Due to 6-fold symmetry, we can restrict phi to [0, pi/6]
            # and use the fact that the support function is piecewise cosine.
            
            # Actually, for a regular hexagon with circumradius 1 (side length 1/sqrt(3)? No, side=1 means circumradius=1)
            # Wait, for a regular hexagon with side length s, the circumradius is s.
            # So vertices are at distance 1 from center.
            
            # h(phi) for hexagon with rotation theta:
            # h(phi) = max_k cos(phi - theta - k*pi/3)
            
            # The minimum separation between two hexagons with relative rotation dtheta
            # is min_phi [h1(phi) + h2(phi+pi)] where h2 has rotation dtheta relative to h1.
            # h2(phi+pi) = max_k cos(phi+pi - dtheta - k*pi/3) = max_k cos(phi - dtheta - k*pi/3 + pi)
            #            = max_k -cos(phi - dtheta - k*pi/3)
            #            = -min_k cos(phi - dtheta - k*pi/3)
            #            = -h2(-phi - dtheta) ... this is getting complex.
            
            # Simpler approach: the Minkowski sum of two hexagons is a dodecagon (or hexagon if aligned).
            # The support function of the Minkowski sum is h_sum(phi) = h1(phi) + h2(phi).
            # The minimum distance is min over phi of h_sum(phi) where we consider the "inner" support.
            # Actually, for two convex bodies A and B to be disjoint with centers at distance d,
            # we need d >= max over all directions of (h_A(phi) + h_B(phi+pi)) for the direction connecting centers?
            # No, the condition for non-overlap is that for all directions phi:
            #   h_A(phi) + h_B(phi + pi) <= d
            # where d is the distance between centers.
            # So the minimum required d is max_phi [h_A(phi) + h_B(phi+pi)]
            
            # For hexagon 1 with rotation 0: h1(phi) = max_k cos(phi - k*pi/3)
            # For hexagon 2 with rotation dtheta: h2(phi+pi) = max_k cos(phi+pi - dtheta - k*pi/3)
            
            # Let's compute numerically
            phi_vals = np.linspace(0, 2*math.pi, 720)
            h1 = np.max([np.cos(phi_vals - k*math.pi/3.0) for k in range(6)], axis=0)
            h2 = np.max([np.cos(phi_vals + math.pi - dtheta - k*math.pi/3.0) for k in range(6)], axis=0)
            return np.max(h1 + h2)

        # Precompute separation constraints for pairs
        # We'll use the discrete solution as a starting point and refine with SLSQP
        
        def objective_and_constraints(params):
            """
            params: [x0, y0, theta0, x1, y1, theta1, ..., x11, y11, theta11]
            Returns (objective, constraints)
            """
            n_params = 3 * N
            x = params[0::3]
            y = params[1::3]
            theta = params[2::3]
            
            # Objective: outer side length
            # Compute max projection of all vertices onto each normal
            max_extent = 0.0
            for i in range(N):
                for k in range(6):
                    vx = x[i] + math.cos(theta[i] + k * math.pi / 3.0)
                    vy = y[i] + math.sin(theta[i] + k * math.pi / 3.0)
                    for j in range(6):
                        proj = vx * normal_cos[j] + vy * normal_sin[j]
                        if proj > max_extent:
                            max_extent = proj
            
            L = 2.0 * max_extent / sqrt3
            
            # Constraints: non-overlap between all pairs
            # For each pair (i, j), the distance must be >= min_separation(theta_i - theta_j)
            constraints = []
            for i in range(N):
                for j in range(i + 1, N):
                    dtheta = theta[i] - theta[j]
                    min_sep = min_separation(dtheta)
                    # Constraint: dist(i,j) - min_sep >= 0
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = math.sqrt(dx*dx + dy*dy)
                    constraints.append(dist - min_sep)
            
            return L, constraints
        
        # Use SLSQP to minimize L subject to non-overlap constraints
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[i * 3] = init_positions[i, 0]
            x0[i * 3 + 1] = init_positions[i, 1]
            x0[i * 3 + 2] = init_thetas[i]
        
        # Define constraints for scipy
        def make_constraint(i, j):
            def constraint(params):
                x = params[0::3]
                y = params[1::3]
                theta = params[2::3]
                dtheta = theta[i] - theta[j]
                min_sep = min_separation(dtheta)
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - min_sep
            return constraint
        
        constraints_list = []
        for i in range(N):
            for j in range(i + 1, N):
                constraints_list.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
        # Objective function for scipy
        def scipy_objective(params):
            x = params[0::3]
            y = params[1::3]
            theta = params[2::3]
            
            max_extent = 0.0
            for i in range(N):
                for k in range(6):
                    vx = x[i] + math.cos(theta[i] + k * math.pi / 3.0)
                    vy = y[i] + math.sin(theta[i] + k * math.pi / 3.0)
                    for j in range(6):
                        proj = vx * normal_cos[j] + vy * normal_sin[j]
                        if proj > max_extent:
                            max_extent = proj
            
            return 2.0 * max_extent / sqrt3
        
        try:
            result = minimize(
                scipy_objective,
                x0,
                method='SLSQP',
                constraints=constraints_list,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < best_L:
                best_L = result.fun
                for i in range(N):
                    init_positions[i, 0] = result.x[i * 3]
                    init_positions[i, 1] = result.x[i * 3 + 1]
                    init_thetas[i] = result.x[i * 3 + 2]
        except Exception:
            pass

    # Final validation: check that the solution is valid
    # Verify non-overlap
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dtheta = init_thetas[i] - init_thetas[j]
            min_sep = min_separation(dtheta) if HAS_SCIPY else 2.0  # Fallback: assume need distance >= 2
            dx = init_positions[i, 0] - init_positions[j, 0]
            dy = init_positions[i, 1] - init_positions[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_sep - 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to discrete solution
        for idx, i in enumerate(best_subset):
            x, y, _, _ = all_centers[i]
            rx = x * math.cos(best_cluster_angle) - y * math.sin(best_cluster_angle)
            ry = x * math.sin(best_cluster_angle) + y * math.cos(best_cluster_angle)
            init_positions[idx, 0] = rx * SCALE
            init_positions[idx, 1] = ry * SCALE
            init_thetas[idx] = best_thetas[idx]
        # Recompute best_L
        best_L = min_L_for_hexagon(0, 0, 0)  # Placeholder, will be computed properly
        max_extent = 0.0
        for i in range(N):
            for k in range(6):
                vx = init_positions[i, 0] + math.cos(init_thetas[i] + k * math.pi / 3.0)
                vy = init_positions[i, 1] + math.sin(init_thetas[i] + k * math.pi / 3.0)
                for j in range(6):
                    proj = vx * normal_cos[j] + vy * normal_sin[j]
                    if proj > max_extent:
                        max_extent = proj
        best_L = 2.0 * max_extent / sqrt3

    inner = []
    for i in range(N):
        inner.append([init_positions[i, 0], init_positions[i, 1], math.degrees(init_thetas[i])])

    center = (0.0, 0.0)
    side = best_L
    angle = 0.0  # Outer hexagon angle (flat-top)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
