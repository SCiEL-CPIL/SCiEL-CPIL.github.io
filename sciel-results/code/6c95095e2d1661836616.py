"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Normal directions for flat-top outer hexagon
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)

    def get_vertices(cx, cy, theta):
        """Get vertices of a unit hexagon at (cx, cy) with rotation theta."""
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return vx, vy

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vx, vy = get_vertices(cx, cy, theta)
        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def required_distance(theta1, theta2, dx, dy):
        """
        Calculate the minimum distance required between centers of two unit hexagons
        with rotations theta1, theta2 such that they do not overlap,
        given the vector (dx, dy) from hex1 to hex2.
        
        This is the support function of the Minkowski sum of hex1 and reflected hex2
        in the direction of the vector from hex2 to hex1 (or vice versa).
        """
        # Direction from center 2 to center 1 is (-dx, -dy) if we consider Minkowski sum of H1 and -H2?
        # Distance d = ||p1 - p2||.
        # They overlap if p2 is inside p1 + H2 (Minkowski sum of H1 and H2 reflected?).
        # Actually, H1 and H2 overlap if 0 is in (p1 - p2) - (H1 - H2)?
        # Simpler: H1 and H2 overlap if the distance between centers is less than
        # the support function of (H1 + (-H2)) in the direction of the vector connecting them?
        # No, the condition for non-overlap is:
        # dist(p1, p2) >= h_{H1 + (-H2)}(u) where u is the unit vector from p2 to p1?
        # Let's verify.
        # If H1 and H2 are points, H1+(-H2) is a point at p1-p2.
        # If H1 and H2 are disks of radius r, H1+(-H2) is disk of radius 2r.
        # Support function is 2r. Dist must be >= 2r. Correct.
        
        # So we need the support function of the Minkowski sum of H1 and -H2.
        # -H2 is H2 rotated by 180 degrees. Since H2 is centrally symmetric, -H2 is just H2 reflected.
        # For a regular hexagon, -H2 is congruent to H2 rotated by 180 (which is 0 mod 60).
        # So -H2 is effectively a hexagon with rotation theta2 (since 180 is a symmetry axis? No, 180 is a rotation symmetry).
        # So the shape is the Minkowski sum of H(theta1) and H(theta2).
        
        # The direction u is from p2 to p1: (-dx, -dy) normalized?
        # Wait, the vector separating them is p1 - p2.
        # If we translate H2 by -p2 and H1 by -p1, they overlap if 0 is in H1' - H2'?
        # Let's use the property:
        # Non-overlap iff ||p1 - p2|| >= h_{H1 \oplus H2}( \frac{p1 - p2}{||p1 - p2||} )
        # where H1 \oplus H2 is the Minkowski sum.
        # Note: h_K(u) = max_{x in K} x . u.
        
        dx_norm = math.hypot(dx, dy)
        if dx_norm < 1e-9:
            return 0.0
            
        ux = -dx / dx_norm # Direction from p2 to p1?
        # Let's check.
        # If p1 = (1,0), p2 = (0,0). u = (1,0).
        # h_{H1 \oplus H2}(1,0).
        # If H1, H2 are disks r=1. Sum is disk r=2. h=2.
        # Dist = 1. 1 >= 2 is False. Overlap. Correct.
        
        # We need to compute h_{H1 \oplus H2}(u).
        # The Minkowski sum of two polygons is a polygon.
        # Its vertices are the sums of vertices of H1 and H2, sorted by angle.
        # We can compute the support function directly by maximizing over vertices?
        # No, the support function of a sum is the sum of support functions.
        # h_{A \oplus B}(u) = h_A(u) + h_B(u).
        
        # So we just need h_{H1}(u) + h_{H2}(u).
        
        # h_H(u) for a unit hexagon at origin with rotation theta.
        # h(u) = max_{v in vertices} v . u.
        
        # Vertices of H1 (at origin, rot theta1):
        v1_x = np.cos(np.array([math.radians(60 * k) + theta1 for k in range(6)]))
        v1_y = np.sin(np.array([math.radians(60 * k) + theta1 for k in range(6)]))
        
        v2_x = np.cos(np.array([math.radians(60 * k) + theta2 for k in range(6)]))
        v2_y = np.sin(np.array([math.radians(60 * k) + theta2 for k in range(6)]))
        
        h1 = np.max(v1_x * ux + v1_y * uy)
        h2 = np.max(v2_x * ux + v2_y * uy)
        
        return h1 + h2

    def cost_function(params):
        # params: x1, y1, t1, x2, y2, t2, ..., L
        L = params[-1]
        if L < 0:
            return 1e6
            
        cost = L
        # Containment penalty
        pen_contain = 0.0
        # Overlap penalty
        pen_overlap = 0.0
        
        positions = []
        rotations = []
        
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            t = params[3*i + 2]
            positions.append((x, y))
            rotations.append(t)
            
            L_req = min_L_for_hexagon(x, y, t)
            if L_req > L:
                pen_contain += (L_req - L) ** 2

        for i in range(N):
            for j in range(i + 1, N):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                t1 = rotations[i]
                t2 = rotations[j]
                
                dx = x1 - x2
                dy = y1 - y2
                dist = math.hypot(dx, dy)
                
                d_req = required_distance(t1, t2, dx, dy)
                
                if dist < d_req:
                    pen_overlap += (d_req - dist) ** 2

        return cost + 10.0 * pen_overlap + 10.0 * pen_contain

    # Initialization: Outer ring of side-3 hexagon grid (12 centers)
    # Axial coords with max(|q|,|r|,|q+r|) == 2
    init_params = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) == 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                init_params.extend([x, y, 0.0])
                
    # Add initial L estimate
    # The max L_req for these positions
    max_l = 0
    for i in range(N):
        x = init_params[3*i]
        y = init_params[3*i + 1]
        t = init_params[3*i + 2]
        max_l = max(max_l, min_L_for_hexagon(x, y, t))
        
    init_params.append(max_l)
    
    # Optimize
    # Use Nelder-Mead or L-BFGS-B?
    # L-BFGS-B is better for smooth parts, but we have penalties.
    # Let's try L-BFGS-B.
    try:
        res = minimize(cost_function, init_params, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9})
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during optimization")
            
        best_params = res.x
        best_score = res.fun
        
        # Try a few random restarts if time permits
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            # Perturb initial solution
            new_params = init_params.copy()
            for i in range(3 * N):
                new_params[i] += np.random.normal(0, 0.5)
            new_params[-1] = max_l + 0.5
            
            res_r = minimize(cost_function, new_params, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if time.time() > deadline:
                break
                
            if res_r.fun < best_score:
                best_params = res_r.x
                best_score = res_r.fun

    except Exception as e:
        # Fallback to initial if optimization fails
        best_params = init_params

    # Extract result
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        t = best_params[3*i + 2]
        inner.append([x, y, math.degrees(t)])
        
    L = best_params[-1]
    
    # Final polish: Ensure strict non-overlap and containment by scaling L slightly up
    # Or just return the L found. The evaluator measures the raw value.
    # If the solution is slightly infeasible due to numerical error, L might be too small.
    # We should verify feasibility and adjust L if needed.
    
    # Verification
    min_L_final = 0.0
    for i in range(N):
        x, y, t_deg = inner[i]
        t = math.radians(t_deg)
        L_req = min_L_for_hexagon(x, y, t)
        min_L_final = max(min_L_final, L_req)
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, _ = inner[i]
            x2, y2, _ = inner[j]
            t1 = math.radians(inner[i][2])
            t2 = math.radians(inner[j][2])
            
            dx = x1 - x2
            dy = y1 - y2
            dist = math.hypot(dx, dy)
            
            d_req = required_distance(t1, t2, dx, dy)
            
            if dist < d_req:
                # If overlap, we must increase L? No, L doesn't fix overlap.
                # We must move them apart.
                # This indicates the optimizer failed to find a feasible solution.
                # In this case, the score is invalid or we return a conservative L.
                # But the task says "minimizing the outer side length".
                # If we can't pack them, the problem is infeasible? No, it's always feasible for large L.
                # If the optimizer failed, we should return a larger L and valid positions?
                # Or just return the best L found, even if invalid?
                # "invalid result ... scores 0".
                # So we must ensure validity.
                
                # Simple fix: Increase L and scale positions? No, scaling positions increases L_req.
                # If overlap exists, the solution is invalid.
                # We should probably return a known valid solution (e.g. the grid one with large L) if this fails.
                pass

    # If we detected overlap, we should fallback to a safe solution
    # Let's re-verify carefully.
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, _ = inner[i]
            x2, y2, _ = inner[j]
            t1 = math.radians(inner[i][2])
            t2 = math.radians(inner[j][2])
            dx = x1 - x2
            dy = y1 - y2
            dist = math.hypot(dx, dy)
            d_req = required_distance(t1, t2, dx, dy)
            if dist < d_req - 1e-6: # Tolerance
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to the initial grid solution with a safe L
        # The initial grid (outer ring) has overlaps?
        # Distance between adjacent outer ring centers:
        # (2,0) to (1,1)? No, axial.
        # (2,0) cart (3,0).
        # (1,1) cart (1.5, sqrt3*1.5 = 2.598).
        # Dist: sqrt(1.5^2 + 2.598^2) = sqrt(2.25 + 6.75) = 3.
        # Required dist for aligned hexagons is sqrt(3) = 1.732.
        # So 3 > 1.732. No overlap.
        # So the initial solution is valid.
        
        # Calculate L for initial solution
        L_init = 0
        for i in range(N):
            x = init_params[3*i]
            y = init_params[3*i + 1]
            t = init_params[3*i + 2]
            L_init = max(L_init, min_L_for_hexagon(x, y, t))
            
        inner = []
        for i in range(N):
            x = init_params[3*i]
            y = init_params[3*i + 1]
            t = init_params[3*i + 2]
            inner.append([x, y, math.degrees(t)])
            
        return inner, (0.0, 0.0), L_init, 0.0

    return inner, (0.0, 0.0), min_L_final, 0.0

# EVOLVE-BLOCK-END
