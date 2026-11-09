"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
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

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
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
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine around best
        for _ in range(2):
            step = 0.1 if _ == 0 else 0.01
            for t_deg in np.arange(best_theta - step*10, best_theta + step*10 + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap
    SCALE = 1.000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But the optimal might be at 30 degrees (rotating the "corners" to align with "sides")
    cluster_angles_to_try = []
    # Coarse search
    for a_deg in range(0, 61, 3):
        cluster_angles_to_try.append(math.radians(a_deg))
    # Add 30 degrees explicitly (often optimal)
    cluster_angles_to_try.append(math.radians(30))

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

        # Try all subsets of size 12 from 19 centers
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
        def smooth_objective(params):
            """
            params: [x0, y0, theta0, x1, y1, theta1, ..., x11, y11, theta11, L]
            Minimize L + penalty for overlaps
            """
            L = params[-1]
            positions = params[:2*N].reshape(N, 2)
            thetas = params[2*N:3*N]
            
            # Compute required L for each hexagon
            L_required = np.zeros(N)
            for i in range(N):
                L_required[i] = min_L_for_hexagon(positions[i, 0], positions[i, 1], thetas[i])
            
            # Soft-max approximation of max(L_required)
            # Use log-sum-exp for smoothness
            beta = 50.0  # Higher = sharper max approximation
            max_lr = np.max(L_required)
            if max_lr > 0:
                soft_max = max_lr + math.log(np.sum(np.exp(beta * (L_required - max_lr)))) / beta
            else:
                soft_max = 0.0
            
            # Overlap penalty: for each pair, check if hexagons overlap
            # Two unit hexagons with centers distance d apart and relative angle delta_theta
            # They overlap if d < some function of delta_theta
            # Approximate: minimum distance for non-overlap depends on relative rotation
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    
                    # Relative rotation
                    dtheta = abs(thetas[i] - thetas[j]) % (2 * math.pi)
                    dtheta = min(dtheta, 2*math.pi - dtheta)
                    
                    # Minimum separation for non-overlap
                    # For aligned hexagons (dtheta=0), min distance = 2 * apothem = sqrt(3)
                    # For 30 degree offset, min distance is larger
                    # Approximate: min_dist = sqrt(3) * (1 + 0.1 * (dtheta / (math.pi/6)))
                    min_dist = sqrt3 * (1.0 + 0.15 * (dtheta / (math.pi/6)))
                    
                    if d < min_dist:
                        # Smooth penalty
                        deficit = min_dist - d
                        overlap_penalty += deficit * deficit * 100.0
            
            # Also penalize if L is too small to contain all hexagons
            containment_penalty = 0.0
            for i in range(N):
                if L_required[i] > L:
                    deficit = L_required[i] - L
                    containment_penalty += deficit * deficit * 100.0
            
            return L + overlap_penalty + containment_penalty
        
        # Initial guess for L
        L_init = best_L * 1.01
        init_params = np.concatenate([init_positions.flatten(), init_thetas, [L_init]])
        
        # Try multiple optimization methods
        best_opt_L = float('inf')
        best_opt_params = None
        
        methods_to_try = ['Nelder-Mead', 'Powell']
        
        for method in methods_to_try:
            if time.time() > deadline - 5.0:
                break
            
            try:
                if method == 'Nelder-Mead':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                elif method == 'Powell':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Powell',
                        options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-8}
                    )
                
                if result.success or result.fun < smooth_objective(init_params):
                    opt_L = result.x[-1]
                    # Verify this is valid by checking actual constraints
                    opt_positions = result.x[:2*N].reshape(N, 2)
                    opt_thetas = result.x[2*N:3*N]
                    
                    # Check actual max L required
                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon(opt_positions[i,0], opt_positions[i,1], opt_thetas[i]))
                    
                    # Check overlaps
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = opt_positions[i, 0] - opt_positions[j, 0]
                            dy = opt_positions[i, 1] - opt_positions[j, 1]
                            d = math.sqrt(dx*dx + dy*dy)
                            dtheta = abs(opt_thetas[i] - opt_thetas[j]) % (2 * math.pi)
                            dtheta = min(dtheta, 2*math.pi - dtheta)
                            min_dist = sqrt3 * (1.0 + 0.15 * (dtheta / (math.pi/6)))
                            if d < min_dist - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and actual_max_L < best_opt_L:
                        best_opt_L = actual_max_L
                        best_opt_params = result.x.copy()
                        
            except Exception:
                pass
        
        # If optimization found a better solution, use it
        if best_opt_params is not None and best_opt_L < best_L:
            best_L = best_opt_L
            for i in range(N):
                init_positions[i, 0] = best_opt_params[2*i]
                init_positions[i, 1] = best_opt_params[2*i+1]
                init_thetas[i] = best_opt_params[2*N+i]

    # Construct inner hexagons
    inner = []
    for idx in range(N):
        inner.append([init_positions[idx, 0], init_positions[idx, 1], math.degrees(init_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
