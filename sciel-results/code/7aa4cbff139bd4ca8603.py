"""Packing 12 unit hexagons in minimal regular hexagon.
Strategy: Use a grid of candidate positions, try all 12-subsets, and for each subset
compute the optimal rotation of each hexagon to minimize the outer hexagon side length.
Then use local optimization to refine positions and rotations."""
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
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = SQRT3

    # Generate candidate centers on a hexagonal grid
    # For a hexagon of side L, the hexagonal grid spacing is 2*sqrt(3) for non-overlapping
    # But we can try denser grids. Let's use a grid with spacing ~1.7 (slightly less than 2*sqrt(3)≈3.46)
    # Actually, unit hexagons need center distance >= 2*sqrt(3) ≈ 3.464 to not overlap (for aligned hexagons)
    # But with rotation, they can be closer. Let's use a grid with spacing 1.8 initially.
    
    # Better approach: generate a hexagonal lattice of points
    # For flat-top hexagons, the lattice vectors are:
    # v1 = (3, 0) and v2 = (1.5, 3*sqrt(3)/2) for unit hexagons (side 1)
    # Actually, the center-to-center distance for adjacent hexagons in a hex grid is 2*sqrt(3)
    # Let's generate points on a finer grid
    
    # Generate a hexagonal grid of candidate positions
    # The hexagonal grid for unit hexagons has spacing 2*sqrt(3) between centers
    # Let's use a grid with spacing 1.5 (finer) to allow for rotated hexagons
    
    grid_spacing = 1.5
    extent = 12.0  # Search within this range
    
    # Generate hexagonal grid points
    # For a hexagonal grid, we can use axial coordinates
    # q, r -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    # This gives spacing of 2*sqrt(3) between adjacent centers for unit hexagons
    
    # Let's use a denser grid
    all_centers = []
    # Use a range of q and r
    q_range = range(-8, 9)
    for q in q_range:
        for r in range(-8, 9):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            if abs(x) <= extent and abs(y) <= extent:
                all_centers.append((x, y))
    
    # Also add a finer grid
    fine_spacing = 0.75
    for i in range(-16, 17):
        for j in range(-16, 17):
            x = fine_spacing * i
            y = fine_spacing * j
            if abs(x) <= extent and abs(y) <= extent:
                all_centers.append((x, y))
    
    # Remove duplicates (within tolerance)
    unique_centers = []
    for c in all_centers:
        is_dup = False
        for uc in unique_centers:
            if abs(c[0] - uc[0]) < 0.1 and abs(c[1] - uc[1]) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_centers.append(c)
    
    all_centers = unique_centers
    n_centers = len(all_centers)
    
    print(f"Number of candidate centers: {n_centers}")
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # The outer hexagon has sides at angles 30, 90, 150, 210, 270, 330 degrees
    # The normals point in these directions
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 1 degree steps
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = max(0.0, best_theta - step * 10)
            t_end = min(math.radians(60), best_theta + step * 10)
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end) + step_deg, step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Small scale to ensure strict disjointness
    SCALE = 1.000001
    
    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_positions = None
    
    # Precompute optimal theta for each center
    center_optimal = []
    for i in range(n_centers):
        x, y = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
        center_optimal.append((L_h, theta_h))
    
    # Sort centers by their individual L requirement
    sorted_indices = sorted(range(n_centers), key=lambda i: center_optimal[i][0])
    
    # Try all subsets of size 12 from the top candidates
    # To limit the search, only consider the top 25 centers
    top_k = 25
    top_centers = sorted_indices[:top_k]
    
    print(f"Trying subsets from top {top_k} centers...")
    
    for subset in combinations(top_centers, N):
        if time.time() > deadline - 20.0:
            break
        L = max(center_optimal[i][0] for i in subset)
        if L < best_L:
            best_L = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
            best_positions = [all_centers[i] for i in subset]
            # Early termination if we find a very good solution
            if best_L < 9.95:
                break
    
    if best_subset is None:
        # Fallback: use first 12 centers
        best_subset = tuple(sorted_indices[:N])
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_positions = [all_centers[i] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)
    
    # Build initial positions from best discrete solution
    init_positions = np.array([(p[0] * SCALE, p[1] * SCALE) for p in best_positions])
    init_thetas = np.array(best_thetas)
    
    print(f"Initial L: {best_L}")
    
    # Smooth optimization using scipy if available
    if HAS_SCIPY and time.time() < deadline - 15.0:
        # Define the objective: minimize the maximum L needed
        def compute_L(positions, thetas):
            """Compute the outer side length L for given positions and rotations."""
            max_L = 0.0
            for i in range(N):
                cx, cy = positions[i]
                theta = thetas[i]
                L_i = min_L_for_hexagon(cx, cy, theta)
                if L_i > max_L:
                    max_L = L_i
            return max_L
        
        # Non-overlap check using support functions
        def hex_support(cx, cy, theta, phi):
            """Support function of a unit hexagon at (cx,cy) with rotation theta in direction phi."""
            max_val = 0.0
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                proj = vx * math.cos(phi) + vy * math.sin(phi)
                if proj > max_val:
                    max_val = proj
            return max_val
        
        def min_separation(theta1, theta2):
            """
            Compute the minimum center-to-center distance for two unit hexagons
            with rotations theta1 and theta2 to be non-overlapping.
            Uses the Minkowski sum approach.
            """
            min_sep = float('inf')
            
            # Check all 12 critical directions (6 from each hexagon)
            for k in range(6):
                phi1 = theta1 + k * math.pi / 3.0
                phi2 = theta2 + k * math.pi / 3.0
                
                for phi in [phi1, phi2]:
                    # Support of hex1 in direction phi
                    h1 = 0.0
                    for j in range(6):
                        angle = theta1 + j * math.pi / 3.0
                        proj = math.cos(angle) * math.cos(phi) + math.sin(angle) * math.sin(phi)
                        if proj > h1:
                            h1 = proj
                    
                    # Support of hex2 in direction -phi (i.e., phi + pi)
                    h2 = 0.0
                    for j in range(6):
                        angle = theta2 + j * math.pi / 3.0
                        proj = math.cos(angle) * math.cos(phi + math.pi) + math.sin(angle) * math.sin(phi + math.pi)
                        if proj > h2:
                            h2 = proj
                    
                    sep = h1 + h2
                    if sep < min_sep:
                        min_sep = sep
            
            return min_sep
        
        def overlap_violation(positions, thetas):
            """Compute the maximum overlap violation between any pair of hexagons."""
            max_violation = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    min_sep = min_separation(thetas[i], thetas[j])
                    violation = min_sep - dist
                    if violation > max_violation:
                        max_violation = violation
            return max_violation
        
        # Objective: minimize L + penalty for overlaps
        def objective(params):
            positions = params[:2 * N].reshape(N, 2)
            thetas = params[2 * N:]
            
            L = compute_L(positions, thetas)
            ov = overlap_violation(positions, thetas)
            
            # Penalize overlaps heavily
            penalty = 100.0 * max(0.0, ov)
            return L + penalty
        
        # Initial parameters
        x0 = np.concatenate([init_positions.flatten(), init_thetas])
        
        # Run optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < best_L:
                opt_params = result.x
                opt_positions = opt_params[:2 * N].reshape(N, 2)
                opt_thetas = opt_params[2 * N:]
                
                # Verify no overlaps
                ov = overlap_violation(opt_positions, opt_thetas)
                if ov < 1e-6:
                    opt_L = compute_L(opt_positions, opt_thetas)
                    if opt_L < best_L:
                        best_L = opt_L
                        init_positions = opt_positions
                        init_thetas = opt_thetas
        except Exception as e:
            print(f"Optimization error: {e}")
    
    # Final high-precision polish: refine thetas with very fine steps
    final_positions = init_positions.copy()
    final_thetas = init_thetas.copy()
    
    # Do one more pass of theta refinement for each hexagon
    for i in range(N):
        if time.time() > deadline - 5.0:
            break
        cx, cy = final_positions[i]
        best_theta_i = final_thetas[i]
        best_L_i = min_L_for_hexagon(cx, cy, best_theta_i)
        
        # Very fine search
        for t_deg in np.arange(math.degrees(best_theta_i) - 0.01, math.degrees(best_theta_i) + 0.011, 0.001):
            t = math.radians(t_deg)
            L_i = min_L_for_hexagon(cx, cy, t)
            if L_i < best_L_i:
                best_L_i = L_i
                best_theta_i = t
        
        final_thetas[i] = best_theta_i
    
    # Compute final L
    final_L = 0.0
    for i in range(N):
        L_i = min_L_for_hexagon(final_positions[i, 0], final_positions[i, 1], final_thetas[i])
        if L_i > final_L:
            final_L = L_i
    
    # Construct inner hexagons
    inner = []
    for i in range(N):
        inner.append([final_positions[i, 0], final_positions[i, 1], math.degrees(final_thetas[i])])
    
    return inner, (0.0, 0.0), final_L, 0.0
