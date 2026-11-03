"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # 1. Generate initial honeycomb configuration
    # A regular hexagon of side 1 has "radius" (center to vertex) = 1.
    # Distance between centers in a honeycomb packing is 2.
    # We arrange 12 hexagons in a cluster.
    # Center at (0,0).
    # 1st ring: 6 hexagons at distance 2, angles 0, 60, ..., 300.
    # 2nd ring: We need 5 more. Place them at distance 4? Or fill gaps?
    # Let's just place them in a spiral or grid initially, then let optimizer fix it.
    # A grid of 3x4 or similar might be better for initialization than strict honeycomb if we allow rotation.
    # But honeycomb is the densest.
    # Let's create a list of 12 positions.
    
    # Positions in a hex grid (axial coordinates converted to Cartesian)
    # q, r -> x = 2*q + r, y = sqrt(3)*r (scaled by side length 1? No, distance 2 is side*2)
    # For side 1, distance between centers is 2.
    # x = 2 * q + r
    # y = sqrt(3) * r
    
    coords = []
    # Center
    coords.append((0, 0))
    # Ring 1 (6 hexes)
    for i in range(6):
        q, r = [ (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) ][i]
        x = 2 * q + r
        y = math.sqrt(3) * r
        coords.append((x, y))
    # Ring 2 (need 5 more to make 12). 
    # Ring 2 has 12 positions. We pick 5 adjacent to Ring 1 to keep it compact.
    # Let's pick: (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ... wait, let's just pick 5 from the outer ring.
    # Actually, a 3x4 rectangular grid of hexes is a valid starting point that the optimizer can rotate/translate.
    # Let's stick to a compact cluster.
    # Let's add 5 more from the second ring.
    ring2_coords = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1) # Just some valid axial coords in ring 2
    ]
    # Convert to Cartesian
    for q, r in ring2_coords:
        x = 2 * q + r
        y = math.sqrt(3) * r
        coords.append((x, y))
        
    # Ensure we have exactly 12
    coords = coords[:12]
    
    # Initial angles: 0
    initial_positions = np.array(coords)
    initial_angles = np.zeros(N)
    
    # Outer hexagon initial guess
    # The cluster roughly has radius ~4. Outer side length S.
    # Distance from center to vertex of outer hex is S.
    # So S should be at least max distance of inner vertices from outer center.
    # Inner max dist ~ 4 + 1 (radius of inner) = 5.
    # So S ~ 5.
    outer_center = np.array([0.0, 0.0])
    outer_side = 6.0
    outer_angle = 0.0
    
    # Total variables: 2*N (pos) + N (angle) + 2 (outer center) + 1 (outer side) + 1 (outer angle)
    # 24 + 12 + 2 + 1 + 1 = 40 variables.
    
    def flatten_params(x, y, ang, oc, os, oa):
        return np.concatenate([
            x.flatten(), y.flatten(), ang, 
            [oc[0], oc[1]], [os], [oa]
        ])
    
    def unpack_params(v):
        x = v[0:N]
        y = v[N:2*N]
        ang = v[2*N:3*N]
        oc = v[3*N:3*N+2]
        os = v[3*N+2]
        oa = v[3*N+3]
        return x, y, ang, oc, os, oa

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append([vx, vy])
        return np.array(verts)

    def cost_function(v):
        if time.time() > deadline:
            return 1e6 # Penalty for timeout
        
        x, y, ang, oc, os, oa = unpack_params(v)
        
        # Ensure side length is positive
        if os < 0.1:
            return 1e6
            
        # 1. Containment Penalty
        # For each inner hexagon, all vertices must be inside the outer hexagon.
        # Outer hexagon vertices:
        outer_verts = get_hex_vertices(oc[0], oc[1], os, oa)
        
        # A point (px, py) is inside a convex polygon if it is on the same side of all edges.
        # For a regular hexagon centered at oc, rotated by oa:
        # The distance from center to edge (apothem) is os * cos(pi/6) = os * sqrt(3)/2.
        # The normal vectors to the edges point outwards.
        # Let's use the property: A point p is inside a regular hexagon with center c, side s, angle a
        # if the projection of (p-c) onto the 6 normal directions is <= apothem.
        
        apothem = os * math.sqrt(3) / 2.0
        base_angle = math.radians(oa)
        
        max_violation = 0.0
        
        for i in range(N):
            inner_verts = get_hex_vertices(x[i], y[i], 1.0, ang[i])
            for pv in inner_verts:
                vec = pv - oc
                # Check against 6 faces
                # Face normals are at angles base_angle + i*pi/3 + pi/6 ? 
                # Vertices are at base_angle + i*pi/3.
                # Edges connect V_i and V_{i+1}.
                # Normal to edge i is at angle base_angle + i*pi/3 + pi/6.
                for k in range(6):
                    normal_angle = base_angle + k * math.pi / 3 + math.pi / 6
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    dist = vec[0] * nx + vec[1] * ny
                    # dist must be <= apothem
                    violation = dist - apothem
                    if violation > max_violation:
                        max_violation = violation
        
        # Smooth max violation for containment
        # Use log-sum-exp or just max if we want simplicity, but smooth is better for gradient.
        # Since we iterate over all vertices, max_violation is the worst case.
        # To make it differentiable, we can sum exp(violation) but it's expensive.
        # Let's just use the max violation as a hard constraint penalty? 
        # No, gradient is 0 almost everywhere.
        # Let's use a soft max: M = log(sum(exp(v_i / T)) * T). 
        # But we only tracked the max. Let's re-calculate with a list.
        
        # Optimization: Instead of tracking max, accumulate a soft penalty.
        # But for 12 hexes * 6 verts * 6 faces = 432 checks. It's fast.
        
        # Let's refine the cost to be smooth.
        # We'll compute the vector of violations for all inner vertices against all outer faces.
        
        # Re-do containment check to collect violations
        violations = []
        for i in range(N):
            inner_verts = get_hex_vertices(x[i], y[i], 1.0, ang[i])
            for pv in inner_verts:
                vec = pv - oc
                for k in range(6):
                    normal_angle = base_angle + k * math.pi / 3 + math.pi / 6
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    dist = vec[0] * nx + vec[1] * ny
                    violations.append(dist - apothem)
        
        violations = np.array(violations)
        # Soft max of violations
        # If all violations are negative, soft max is negative (good).
        # We want to minimize the maximum violation.
        # Penalty = soft_max(violations)
        # If max violation < 0, penalty is small.
        # Let's use: C_contain = max(0, soft_max(violations)) ** 2 ? 
        # Or just soft_max(violations).
        
        T = 0.1
        v_max = np.max(violations)
        # log-sum-exp trick
        # sum_exp = np.sum(np.exp((violations - v_max) / T))
        # soft_max = v_max + T * np.log(sum_exp)
        # This is smooth.
        
        # 2. Disjointness Penalty
        # For each pair of inner hexagons, ensure they don't overlap.
        # Distance between centers d.
        # If they are parallel (angle diff 0), min dist is 2.
        # If rotated, it's complex.
        # Approximation: The "radius" of a hexagon is 1 (center to vertex).
        # If distance between centers > 2, they are definitely disjoint.
        # If < 2, they might overlap.
        # A better metric: The distance between the two polygons.
        # For optimization, we can use the distance between centers minus 2 as a proxy?
        # No, rotation matters.
        # Let's use the fact that the minimum distance between two convex polygons is 0 if they touch.
        # We can approximate the "effective radius" of a hexagon at a specific angle.
        # Or simply: penalty = sum over pairs of max(0, (2 - dist_centers) ** 2) ?
        # This assumes they are always "aligned" such that 2 is the min dist.
        # If one rotates 30 deg, the min dist is still 2 (vertex to edge midpoint? No).
        # Vertex to vertex is 2. Edge to edge is sqrt(3) ~ 1.732?
        # Wait. Two unit hexagons.
        # If aligned (flat top), distance between centers for touch is 2 (vertex to vertex) or sqrt(3) (edge to edge)?
        # Flat top:
        #   ___
        #  /   \
        #  \___/
        # Distance between parallel edges is sqrt(3).
        # Distance between vertices is 2.
        # So if they are aligned flat-top, they touch at edges with dist sqrt(3).
        # If rotated 30 deg (pointy top), they touch at vertices with dist 2.
        # So the minimum distance between centers for non-overlap depends on relative rotation.
        # Min dist is sqrt(3) (when edges are parallel and facing each other).
        # Max dist is 2 (when vertices touch).
        # So if dist < sqrt(3), they definitely overlap?
        # No, if dist < sqrt(3), they overlap.
        # If dist > 2, they are definitely disjoint.
        # In between, it depends on rotation.
        
        # For simplicity and speed in the cost function, let's use a conservative bound.
        # Assume min dist is sqrt(3).
        # Penalty if dist < sqrt(3).
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.sqrt(dx*dx + dy*dy)
                min_d = math.sqrt(3) # ~1.732
                if d < min_d:
                    overlap_penalty += (min_d - d) ** 2
        
        # Total Cost
        # We want to minimize outer side length 'os'.
        # But we must satisfy constraints.
        # Objective: os + lambda1 * C_contain + lambda2 * C_overlap
        
        C_contain = v_max + T * np.log(np.sum(np.exp((violations - v_max) / T)))
        # If C_contain is negative, it's good.
        # We only penalize if C_contain > 0?
        # Let's use max(0, C_contain) to allow feasible solutions to have 0 penalty.
        # But max is not smooth.
        # Use softplus: log(1 + exp(C_contain/T2)) * T2
        
        T2 = 0.1
        C_contain_smooth = T2 * np.log(1 + np.exp(C_contain / T2))
        
        total_cost = os + 100.0 * C_contain_smooth + 10.0 * overlap_penalty
        
        return total_cost

    # Initial vector
    v0 = flatten_params(initial_positions[:, 0], initial_positions[:, 1], initial_angles, outer_center, outer_side, outer_angle)
    
    # Optimize
    best_v = v0
    best_cost = cost_function(v0)
    
    # Use L-BFGS-B or Nelder-Mead?
    # L-BFGS-B is faster for smooth problems.
    # Nelder-Mead is more robust to non-smoothness but slower.
    # Our cost is smooth (log-sum-exp).
    
    options = {
        'maxiter': 1000,
        'ftol': 1e-8,
        'gtol': 1e-8
    }
    
    try:
        res = minimize(cost_function, v0, method='L-BFGS-B', options=options)
        if res.fun < best_cost:
            best_cost = res.fun
            best_v = res.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead for local refinement
    if time.time() < deadline - 10:
        try:
            res2 = minimize(cost_function, best_v, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_v = res2.x
        except Exception:
            pass
            
    # Final Polish
    if time.time() < deadline - 5:
        try:
            res3 = minimize(cost_function, best_v, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res3.fun < best_cost:
                best_cost = res3.fun
                best_v = res3.x
        except Exception:
            pass

    x, y, ang, oc, os, oa = unpack_params(best_v)
    
    # Ensure side length is reasonable
    if os < 1.0:
        os = 1.0
        
    inner = [[x[i], y[i], ang[i]] for i in range(N)]
    return inner, (oc[0], oc[1]), os, oa
# EVOLVE-BLOCK-END
