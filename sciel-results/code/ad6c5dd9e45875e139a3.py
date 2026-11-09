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

# Precompute hexagon vertices for various rotations
def hexagon_vertices(cx, cy, theta, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation theta."""
    angles = [math.radians(60 * k) + theta for k in range(6)]
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def polygon_overlap(v1, v2, tol=1e-10):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    # Collect all edge normals
    normals = []
    for verts in [v1, v2]:
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            # Normal perpendicular to edge
            nx = -dy
            ny = dx
            length = math.sqrt(nx * nx + ny * ny)
            if length > 1e-15:
                normals.append((nx / length, ny / length))
    
    # Project all vertices onto each normal
    for nx, ny in normals:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        
        for x, y in v1:
            proj = x * nx + y * ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        
        for x, y in v2:
            proj = x * nx + y * ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        
        # If there's a gap, no overlap
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    
    return True

def exact_min_L_for_hexagon(cx, cy, theta, outer_center=(0, 0), outer_rotation=0.0):
    """
    Exact minimum L for a unit hexagon at (cx,cy) with rotation theta
    to fit inside a regular hexagon of side L centered at outer_center with rotation outer_rotation.
    
    The outer hexagon has vertices at distance L from center in 6 directions.
    For a regular hexagon of side L, the inradius is L * sqrt(3)/2.
    The vertices are at distance L from center.
    
    Actually, for a regular hexagon of side length L:
    - The distance from center to each side (apothem) is L * sqrt(3) / 2
    - The distance from center to each vertex is L
    
    A point (px, py) relative to center is inside if for all 6 face normals n_j:
    dot((px, py), n_j) <= apothem = L * sqrt(3) / 2
    
    So L >= max_j dot((px, py), n_j) / (sqrt(3)/2)
    where (px, py) is relative to outer center.
    """
    sqrt3 = math.sqrt(3.0)
    apothem_factor = sqrt3 / 2.0  # apothem = L * apothem_factor
    
    # Outer hexagon face normals (perpendicular to sides, pointing inward)
    # For a regular hexagon with vertex at angle 0, the faces are at angles 30, 90, 150, 210, 270, 330
    # But we need normals pointing toward the center, so they point in the opposite direction
    # Actually, let's define: face j has outward normal at angle (30 + 60*j) degrees
    # A point is inside if dot(point - center, outward_normal_j) <= apothem for all j
    
    outer_normal_angles = [math.radians(30 + 60 * j + outer_rotation) for j in range(6)]
    outer_normals = [(math.cos(a), math.sin(a)) for a in outer_normal_angles]
    
    # Inner hexagon vertices relative to outer center
    inner_verts = hexagon_vertices(cx - outer_center[0], cy - outer_center[1], theta)
    
    max_proj = 0.0
    for nx, ny in outer_normals:
        for px, py in inner_verts:
            proj = px * nx + py * ny
            max_proj = max(max_proj, proj)
    
    return max_proj / apothem_factor

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
        # Coarse search
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Fine search around best with multiple passes
        for step in [1.0, 0.1, 0.01, 0.001]:
            for t_deg in np.arange(max(0, best_theta - step*5), min(math.radians(60), best_theta + step*5) + step, step):
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
        sqrt3 = math.sqrt(3.0)
        
        def min_separation(dtheta):
            """
            Compute the minimum center-to-center distance for two unit hexagons
            with relative rotation dtheta to be non-overlapping.
            Uses support function approach with fine sampling.
            """
            dtheta = abs(dtheta) % (2 * math.pi)
            dtheta = min(dtheta, 2*math.pi - dtheta)
            
            # For two regular hexagons of side 1, the Minkowski sum is a dodecagon-like shape
            # The minimum separation is the minimum over all directions phi of:
            #   h1(phi) + h2(phi + pi)
            # where h is the support function
            
            min_sep = float('inf')
            # Sample 180 directions (every 1 degree) for better accuracy
            for phi_deg in range(0, 180):
                phi = math.radians(phi_deg)
                # Support function of hexagon 1 (rotation 0) in direction phi
                h1 = max(math.cos(phi - math.radians(60*k)) for k in range(6))
                # Support function of hexagon 2 (rotation dtheta) in direction phi+pi
                h2 = max(math.cos(phi + math.pi - dtheta - math.radians(60*k)) for k in range(6))
                sep = h1 + h2
                if sep < min_sep:
                    min_sep = sep
            
            return min_sep
        
        def check_valid_packing(positions, thetas, L, tol=1e-8):
            """Check if packing is valid: all hexagons inside outer hex and no overlaps."""
            # Check containment
            for i in range(N):
                L_req = min_L_for_hexagon(positions[i, 0], positions[i, 1], thetas[i])
                if L_req > L + tol:
                    return False
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    dtheta = abs(thetas[i] - thetas[j]) % (2 * math.pi)
                    dtheta = min(dtheta, 2*math.pi - dtheta)
                    min_dist = min_separation(dtheta)
                    if d < min_dist - tol:
                        return False
            
            return True
        
        def smooth_objective(params, beta=50.0, penalty_weight=100.0):
            """
            params: [x0, y0, theta0, x1, y1, theta1, ..., x11, y11, theta11, L]
            Minimize L + penalty for overlaps and containment violations
            """
            L = params[-1]
            positions = params[:2*N].reshape(N, 2)
            thetas = params[2*N:3*N]
            
            # Compute required L for each hexagon
            L_required = np.zeros(N)
            for i in range(N):
                L_required[i] = min_L_for_hexagon(positions[i, 0], positions[i, 1], thetas[i])
            
            # Soft-max approximation of max(L_required)
            max_lr = np.max(L_required)
            if max_lr > 1e-10:
                soft_max = max_lr + math.log(np.sum(np.exp(beta * (L_required - max_lr)))) / beta
            else:
                soft_max = 0.0
            
            # Overlap penalty
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    
                    dtheta = abs(thetas[i] - thetas[j]) % (2 * math.pi)
                    dtheta = min(dtheta, 2*math.pi - dtheta)
                    min_dist = min_separation(dtheta)
                    
                    if d < min_dist:
                        deficit = min_dist - d
                        overlap_penalty += deficit * deficit * penalty_weight
            
            # Containment penalty
            containment_penalty = 0.0
            for i in range(N):
                if L_required[i] > L:
                    deficit = L_required[i] - L
                    containment_penalty += deficit * deficit * penalty_weight
            
            return soft_max + overlap_penalty + containment_penalty
        
        def optimize_from_init(init_pos, init_theta, init_L, time_limit):
            """Run optimization from a given initial point, return best found."""
            best_L_local = float('inf')
            best_params_local = None
            
            init_params = np.concatenate([init_pos.flatten(), init_theta, [init_L]])
            
            methods_configs = [
                ('Nelder-Mead', {'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10}),
                ('Powell', {'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}),
                ('Nelder-Mead', {'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-12}),
            ]
            
            for method, opts in methods_configs:
                if time.time() > time_limit:
                    break
                
                try:
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method=method,
                        options=opts
                    )
                    
                    opt_positions = result.x[:2*N].reshape(N, 2)
                    opt_thetas = result.x[2*N:3*N]
                    opt_L = result.x[-1]
                    
                    # Compute actual max L required
                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon(opt_positions[i,0], opt_positions[i,1], opt_thetas[i]))
                    
                    # Check validity
                    if check_valid_packing(opt_positions, opt_thetas, actual_max_L, tol=1e-7):
                        if actual_max_L < best_L_local:
                            best_L_local = actual_max_L
                            best_params_local = result.x.copy()
                            
                except Exception:
                    pass
            
            return best_L_local, best_params_local
        
        # Run optimization from the discrete solution
        L_init = best_L * 1.002
        time_limit = deadline - 5.0
        
        best_opt_L, best_opt_params = optimize_from_init(init_positions, init_thetas, L_init, time_limit)
        
        # Also try with slightly perturbed initial conditions to escape local minima
        if time.time() < time_limit - 2.0:
            np.random.seed(42)
            for pert_iter in range(3):
                if time.time() > time_limit:
                    break
                
                pert_positions = init_positions.copy()
                pert_thetas = init_thetas.copy()
                # Small random perturbations
                pert_positions += np.random.randn(N, 2) * 0.1
                pert_thetas += np.random.randn(N) * 0.05
                
                pert_L = best_L * 1.005
                pert_L_local, pert_params_local = optimize_from_init(pert_positions, pert_thetas, pert_L, time_limit)
                
                if pert_params_local is not None and pert_L_local < best_opt_L:
                    best_opt_L = pert_L_local
                    best_opt_params = pert_params_local
        
        # If optimization found a better solution, use it
        if best_opt_params is not None and best_opt_L < best_L:
            best_L = best_opt_L
            for i in range(N):
                init_positions[i, 0] = best_opt_params[2*i]
                init_positions[i, 1] = best_opt_params[2*i+1]
                init_thetas[i] = best_opt_params[2*N+i]
        
        # Final high-precision polish pass if time remains
        if HAS_SCIPY and time.time() < deadline - 2.0 and best_opt_params is not None:
            try:
                polish_params = best_opt_params.copy()
                result = minimize(
                    smooth_objective,
                    polish_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-13}
                )
                
                opt_positions = result.x[:2*N].reshape(N, 2)
                opt_thetas = result.x[2*N:3*N]
                
                actual_max_L = 0.0
                for i in range(N):
                    actual_max_L = max(actual_max_L, min_L_for_hexagon(opt_positions[i,0], opt_positions[i,1], opt_thetas[i]))
                
                if check_valid_packing(opt_positions, opt_thetas, actual_max_L, tol=1e-9):
                    if actual_max_L < best_L:
                        best_L = actual_max_L
                        for i in range(N):
                            init_positions[i, 0] = opt_positions[i, 0]
                            init_positions[i, 1] = opt_positions[i, 1]
                            init_thetas[i] = opt_thetas[i]
            except Exception:
                pass

    # Construct inner hexagons
    inner = []
    for idx in range(N):
        inner.append([init_positions[idx, 0], init_positions[idx, 1], math.degrees(init_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
