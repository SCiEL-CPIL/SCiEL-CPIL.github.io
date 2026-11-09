"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    n_centers = len(all_centers)  # 19

    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
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
        # Coarse search: 0 to 60 degrees in 0.5 degree steps
        for t_deg in np.arange(0, 60.5, 0.5):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end), step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Precompute optimal theta and L for each center
    center_optimal = []
    for i in range(n_centers):
        x, y, _, _ = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x, y)
        center_optimal.append((L_h, theta_h))

    # Sort centers by their individual L requirement
    sorted_indices = sorted(range(n_centers), key=lambda i: center_optimal[i][0])

    # Initial greedy subset
    best_subset = tuple(sorted_indices[:12])
    best_thetas = [center_optimal[i][1] for i in best_subset]
    best_L = max(center_optimal[i][0] for i in best_subset)

    # Swap improvement
    remaining = sorted_indices[12:]
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for out_idx in range(12):
            for in_idx in remaining:
                new_subset = list(best_subset)
                new_subset[out_idx] = in_idx
                new_L = max(center_optimal[i][0] for i in new_subset)
                if new_L < best_L - 1e-10:
                    best_L = new_L
                    best_subset = tuple(new_subset)
                    best_thetas = [center_optimal[i][1] for i in best_subset]
                    remaining = [i for i in range(n_centers) if i not in best_subset]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Now, perform a local optimization pass on the selected subset
    # We adjust the positions slightly to potentially reduce the max L further
    # while maintaining non-overlap.
    # Since the grid is tight, we can try to "shrink" the cluster slightly towards the center
    # or adjust individual rotations more precisely.
    
    # Let's try to optimize the scale factor of the cluster.
    # If we scale the positions by s < 1, the centers get closer.
    # The constraint is that the distance between any two centers must be > sqrt(3) (for unit hexagons).
    # In the grid, the minimum distance is sqrt(3).
    # So we can't scale down without violating constraints UNLESS we also adjust rotations to create gaps?
    # No, rotation doesn't change the center distance.
    # However, if we move centers *off* the grid, we might be able to reduce the outer L.
    
    # Let's try a simple gradient-free optimization on the positions of the 12 hexagons.
    # Objective: Minimize L
    # Constraints:
    # 1. Non-overlap: dist(i, j) > sqrt(3) for all i != j
    # 2. Inside: L >= L_i(theta_i) for all i
    
    # Since we are constrained to the grid topology (likely), maybe we can just refine the theta values
    # simultaneously to minimize the max L.
    
    # Let's use a simple coordinate descent on thetas.
    # Current thetas are optimal for their *isolated* L.
    # But the global L is max(L_i).
    # If we reduce L for the "bottleneck" hexagon, we might increase L for others?
    # No, L_i depends only on (x_i, y_i, theta_i).
    # So minimizing max(L_i) is just minimizing the maximum of independent functions.
    # The optimal theta for each is the one that minimizes L_i.
    # So the current thetas are already optimal for the given positions.
    
    # The only way to improve is to change positions.
    # Can we move centers?
    # The grid is a rigid structure. If we move a center, we break the symmetry and might allow a smaller L.
    
    # Let's try to move the "corner" hexagons slightly inward.
    # The corners have the highest L.
    # If we move them towards the center, their L decreases.
    # But we must ensure they don't overlap with neighbors.
    
    # Let's identify the "corner" hexagons in our subset.
    # In a side-3 hex grid, the corners are at distance 2*sqrt(3) from center?
    # Distance from (0,0) to (q,r) is sqrt(3) * sqrt(q^2 + qr + r^2).
    # Max distance is for q=2, r=0 -> sqrt(3)*2 = 2*sqrt(3).
    
    # Let's try to optimize the positions of the 12 hexagons using a simple search.
    # We will perturb the positions slightly and check validity.
    
    # Actually, the "SCALE" trick in the seed program suggests that the grid is too tight.
    # The seed scales UP to ensure non-overlap.
    # This implies the seed's L is slightly larger than the theoretical minimum for the grid.
    # Can we do better?
    
    # What if we use a different subset?
    # We already did greedy + swaps.
    
    # What if we rotate the entire cluster?
    # The outer hexagon is fixed at 0 degrees.
    # If we rotate the inner cluster, the L might change.
    # Let's try rotating the entire set of centers by angle phi.
    
    best_L_rot = float('inf')
    best_phi = 0.0
    
    # We need to recompute L for rotated centers.
    # L_i depends on (x_i cos phi - y_i sin phi, x_i sin phi + y_i cos phi, theta_i + phi)
    # Wait, if we rotate the cluster, the relative orientation of hexagons changes with respect to the outer hexagon.
    # So theta_i should also be rotated by phi.
    
    for phi_deg in np.arange(0, 60, 1.0):
        phi = math.radians(phi_deg)
        c, s = math.cos(phi), math.sin(phi)
        max_L = 0.0
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            x_rot = x * c - y * s
            y_rot = x * s + y * c
            theta_rot = center_optimal[i][1] + phi
            L = min_L_for_hexagon(x_rot, y_rot, theta_rot)
            if L > max_L:
                max_L = L
        if max_L < best_L_rot:
            best_L_rot = max_L
            best_phi = phi
    
    # Refine phi
    for step_deg in [0.1, 0.01, 0.001]:
        step = math.radians(step_deg)
        phi_start = best_phi - step * 10
        phi_end = best_phi + step * 10 + step
        for phi_deg in np.arange(math.degrees(phi_start), math.degrees(phi_end), step_deg):
            phi = math.radians(phi_deg)
            c, s = math.cos(phi), math.sin(phi)
            max_L = 0.0
            for i in best_subset:
                x, y, _, _ = all_centers[i]
                x_rot = x * c - y * s
                y_rot = x * s + y * c
                theta_rot = center_optimal[i][1] + phi
                L = min_L_for_hexagon(x_rot, y_rot, theta_rot)
                if L > max_L:
                    max_L = L
            if max_L < best_L_rot:
                best_L_rot = max_L
                best_phi = phi
    
    if best_L_rot < best_L:
        best_L = best_L_rot
        # Update inner positions
        c, s = math.cos(best_phi), math.sin(best_phi)
        inner = []
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            x_rot = x * c - y * s
            y_rot = x * s + y * c
            theta_rot = center_optimal[i][1] + best_phi
            inner.append([x_rot, y_rot, math.degrees(theta_rot)])
    else:
        inner = []
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            inner.append([x, y, math.degrees(center_optimal[i][1])])

    # Check if we have time to try a few random perturbations to break grid symmetry
    if time.time() < deadline - 5:
        # Try to move the corner hexagons inward
        # Identify corners: those with max distance from origin
        dists = []
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            dists.append((math.sqrt(x*x + y*y), i))
        dists.sort(reverse=True)
        
        # The top 3-4 are likely corners
        # Let's try moving them 1% towards the center
        for move_factor in [0.01, 0.02, 0.03]:
            new_inner = []
            valid = True
            max_L = 0.0
            for i in best_subset:
                x, y, _, _ = all_centers[i]
                dist = math.sqrt(x*x + y*y)
                # If it's a "corner" (large dist), move it
                if dist > 1.9 * sqrt3: # Rough heuristic
                    x_new = x * (1 - move_factor)
                    y_new = y * (1 - move_factor)
                else:
                    x_new = x
                    y_new = y
                theta = center_optimal[i][1]
                L = min_L_for_hexagon(x_new, y_new, theta)
                if L > max_L:
                    max_L = L
                new_inner.append([x_new, y_new, math.degrees(theta)])
            
            # Check non-overlap
            # This is expensive, so only do it for promising L
            if max_L < best_L:
                # Quick overlap check
                ok = True
                for a in range(len(new_inner)):
                    for b in range(a+1, len(new_inner)):
                        dx = new_inner[a][0] - new_inner[b][0]
                        dy = new_inner[a][1] - new_inner[b][1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < sqrt3 - 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_L = max_L
                    inner = new_inner

    return inner, (0.0, 0.0), best_L, 0.0
