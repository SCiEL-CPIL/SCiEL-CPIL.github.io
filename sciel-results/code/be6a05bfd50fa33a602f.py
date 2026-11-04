import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Normal points outward: (sqrt(3)/2, 0.5) -> 30 deg
    # Actually, for flat-top hexagon with vertices at 0, 60..:
    # v0=(1,0), v1=(0.5, sqrt3/2). Edge v0->v1. Normal is perpendicular.
    # Let's just compute normals from vertices to be safe.
    unit_normals_0 = []
    for i in range(6):
        v1 = unit_verts_0[i]
        v2 = unit_verts_0[(i+1)%6]
        edge = v2 - v1
        # Outward normal
        n = np.array([-edge[1], edge[0]])
        # Ensure it points away from origin (dot with v1 should be positive)
        if np.dot(n, v1) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        unit_normals_0.append(n)
    unit_normals_0 = np.array(unit_normals_0)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides: 6 directions
        # Outer hexagon vertices are at angles outer_angle, outer_angle+60, ...
        # The supporting lines (normals) are perpendicular to sides.
        # For a regular hexagon with vertices at phi + k*60, the side normals are at phi + 30 + k*60.
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            rot = np.array([[c, -s], [s, c]])
            verts = unit_verts_0 @ rot.T + np.array([cx, cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        # For a regular hexagon of side length S, the inradius (distance from center to side) is S * sqrt(3) / 2.
        # The projection of vertices onto the normal of a side is exactly the inradius.
        # So, max_proj = inradius = S * sqrt(3) / 2
        # S = 2 * max_proj / sqrt(3)
        return max_proj_all * 2.0 / math.sqrt(3)

    def is_valid(positions, angles, tol=1e-5):
        # Use Minkowski sum logic for fast collision check
        # Two convex polygons P1 and P2 are disjoint if and only if there exists a separating axis.
        # We check all edge normals of P1 and P2.
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                # Get normals for p1 and p2
                # Since they are regular hexagons, we can just use the rotated unit normals
                ang1_rad = math.radians(angles[i])
                ang2_rad = math.radians(angles[j])
                c1, s1 = math.cos(ang1_rad), math.sin(ang1_rad)
                rot1 = np.array([[c1, -s1], [s1, c1]])
                n1 = unit_normals_0 @ rot1.T
                
                c2, s2 = math.cos(ang2_rad), math.sin(ang2_rad)
                rot2 = np.array([[c2, -s2], [s2, c2]])
                n2 = unit_normals_0 @ rot2.T
                
                # All candidate separating axes
                axes = np.vstack([n1, n2])
                
                found_separation = False
                for ax in axes:
                    proj1 = p1 @ ax
                    proj2 = p2 @ ax
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Check if intervals [min1, max1] and [min2, max2] are disjoint
                    if max1 < min2 - tol or max2 < min1 - tol:
                        found_separation = True
                        break
                
                if not found_separation:
                    return False
        return True

    def penalty_function(params, target_side):
        """
        Objective function for optimization.
        Returns 0 if valid and contained, otherwise a positive penalty.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        # Check containment in outer hexagon of side length target_side
        # Centered at (0,0), angle 0 for simplicity (we can fix outer angle to 0 without loss of generality for minimal side, 
        # as rotating the whole system doesn't change relative positions, but we might want to optimize outer angle too?
        # Actually, the problem asks for outer hexagon center, side, angle. 
        # If we fix outer center to (0,0), we just need to find positions that fit in a hex of side S.
        # The minimal S is what we want.
        # We can fix outer angle to 0 because the problem is rotationally symmetric? 
        # Wait, the inner hexagons can rotate. The outer hexagon orientation is a degree of freedom.
        # However, since we can rotate the entire configuration, we can fix the outer hexagon angle to 0.
        # The "angle" returned is just for the evaluator.
        
        outer_angle = 0.0
        current_side = calc_min_outer_side(positions, angles, outer_angle)
        
        # Penalty for exceeding target_side
        if current_side > target_side:
            return current_side - target_side
        
        # Check collisions
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                ang1_rad = math.radians(angles[i])
                ang2_rad = math.radians(angles[j])
                c1, s1 = math.cos(ang1_rad), math.sin(ang1_rad)
                rot1 = np.array([[c1, -s1], [s1, c1]])
                n1 = unit_normals_0 @ rot1.T
                c2, s2 = math.cos(ang2_rad), math.sin(ang2_rad)
                rot2 = np.array([[c2, -s2], [s2, c2]])
                n2 = unit_normals_0 @ rot2.T
                axes = np.vstack([n1, n2])
                
                max_overlap = 0.0
                for ax in axes:
                    proj1 = p1 @ ax
                    proj2 = p2 @ ax
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > max_overlap:
                        max_overlap = overlap
                
                if max_overlap > 0:
                    penalty += max_overlap * max_overlap
        
        return penalty

    # Bisection on the outer side length S
    # Lower bound: The minimal possible side length.
    # A single unit hexagon has side 1. The "width" is 2. The inradius is sqrt(3)/2.
    # If we place them in a line, they need space.
    # Let's start with a reasonable lower bound.
    # Area of unit hexagon = 3*sqrt(3)/2 ~ 2.598.
    # 12 hexagons area = 31.176.
    # Area of outer hexagon side S = 6*sqrt(3)/2 * S^2 = 3*sqrt(3) * S^2.
    # 3*sqrt(3) * S^2 >= 31.176 => S^2 >= 31.176 / 5.196 => S^2 >= 6 => S >= 2.45.
    # So S_min is at least 2.45. Let's use 2.5 as lower bound.
    
    low = 2.5
    high = 10.0  # Upper bound, definitely feasible
    
    best_side = high
    best_params = None
    
    # We will perform a binary search on S.
    # For a given S, we try to find a configuration using Differential Evolution.
    # If DE finds a valid config (penalty ~ 0), then S is feasible.
    
    # To save time, we don't do a full binary search with many DE runs.
    # We do a few iterations of bisection.
    
    # Define bounds for DE
    # Positions: roughly within [-4, 4] x [-4, 4]
    # Angles: [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-4, 4))
        bounds.append((-4, 4))
    for i in range(N):
        bounds.append((0, 360))
    
    # DE strategy
    # We want to find ANY solution for a given S.
    # We use 'best1bin' or similar.
    # Since the problem is highly non-convex, we might need multiple restarts or a long run.
    # Given the 145s limit, we can afford maybe 2-4 DE runs of moderate length.
    
    # Let's try a coarse bisection first.
    # If we find a feasible S, we can try to lower it.
    
    # Actually, a better approach:
    # Run DE with an objective that minimizes the *actual* side length, subject to collision constraints?
    # No, DE doesn't handle constraints well directly without penalty.
    # So we stick to the penalty approach.
    
    # Strategy:
    # 1. Try S = 4.0. If feasible, record it.
    # 2. Try S = 3.5. If feasible, record it.
    # 3. Try S = 3.0. If feasible, record it.
    # 4. Try S = 2.8. If feasible, record it.
    # 5. If we have time, refine the best found.
    
    # This is more robust than strict bisection which might waste time on infeasible regions.
    # We are looking for the *minimum*, so we start high and go down.
    
    test_sides = [5.0, 4.5, 4.0, 3.8, 3.5, 3.3, 3.1, 3.0, 2.9, 2.8]
    
    for S in test_sides:
        if time.time() > deadline - 20:
            break
            
        # Run DE to find a configuration for this S
        # We use a small population to keep it fast, but enough to explore.
        # maxiter is limited by time.
        
        def obj_de(params):
            return penalty_function(params, S)
        
        try:
            # Differential Evolution
            # seed for reproducibility? No, random is fine.
            # 'best1bin' is a good default.
            # We set maxiter based on remaining time? 
            # Let's just run for a fixed number of iterations and check time.
            # popsize=15, maxiter=100 might be too slow.
            # Let's use popsize=10, maxiter=50.
            
            res = differential_evolution(
                obj_de, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-8, 
                polish=False, # Disable polish to save time, we can do a local polish later if needed
                disp=False
            )
            
            if res.fun < 1e-6:
                # Found a valid configuration for side S
                best_side = S
                best_params = res.x
                # We can break if we reach a very low side? 
                # Or continue to try smaller sides.
                # Since we are going down, if we succeed, we keep going.
                
        except Exception as e:
            pass
    
    # If we didn't find a valid config for any S in the list, we return the best effort?
    # Or we fall back to a known valid config?
    # The problem requires a valid config. If DE fails, we should use a heuristic.
    
    if best_params is None:
        # Fallback: Generate a simple valid config
        # Place hexagons in a honeycomb pattern
        sqrt3 = math.sqrt(3)
        positions = []
        angles = []
        
        # Honeycomb lattice vectors
        v1 = np.array([1.5, sqrt3/2])
        v2 = np.array([0, sqrt3])
        
        # Generate points
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        
        pts.sort(key=lambda p: np.dot(p, p))
        selected = np.array(pts[:N])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        for p in selected:
            positions.append([p[0], p[1]])
            angles.append(0.0)
            
        pos_arr = np.array(positions)
        ang_arr = np.array(angles)
        
        # Calculate the side length for this config
        side = calc_min_outer_side(pos_arr, ang_arr, 0.0)
        
        # This config is valid by construction (lattice spacing is sufficient for unit hexagons)
        # Wait, is it? 
        # Distance between centers in honeycomb is sqrt(1.5^2 + (sqrt3/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
        # Inradius of unit hex is sqrt(3)/2 ~ 0.866.
        # Distance between parallel sides is sqrt(3) ~ 1.732.
        # So if centers are sqrt(3) apart, the hexagons touch.
        # Yes, this is a valid tiling.
        
        return [pos for pos in positions], (0.0, 0.0), side, 0.0

    # We found a best_params for best_side.
    # Let's verify it and calculate the exact side length.
    pos = best_params[:2*N].reshape(N, 2)
    ang = best_params[2*N:3*N]
    
    # It's possible the DE solution is slightly invalid due to tolerance.
    # Let's check validity.
    if not is_valid(pos, ang, tol=1e-4):
        # Try to repair? Or just use the fallback?
        # If DE found a low penalty but not zero, it might be slightly overlapping.
        # We can try a few local steps to fix it, or just use the fallback if it's too bad.
        # For now, if it's not valid, we use the fallback honeycomb.
        sqrt3 = math.sqrt(3)
        v1 = np.array([1.5, sqrt3/2])
        v2 = np.array([0, sqrt3])
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        pts.sort(key=lambda p: np.dot(p, p))
        selected = np.array(pts[:N])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        positions = []
        angles = []
        for p in selected:
            positions.append([p[0], p[1]])
            angles.append(0.0)
        pos_arr = np.array(positions)
        ang_arr = np.array(angles)
        side = calc_min_outer_side(pos_arr, ang_arr, 0.0)
        return [pos for pos in positions], (0.0, 0.0), side, 0.0

    # Calculate the true minimal side length for this configuration
    final_side = calc_min_outer_side(pos, ang, 0.0)
    
    # The DE objective used a target S. The actual side might be slightly less than S if the penalty was 0.
    # But we used S as the threshold. 
    # Actually, penalty_function returns 0 if current_side <= S AND no collisions.
    # So the actual side is <= S.
    # We should return the actual calculated side.
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], ang[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
