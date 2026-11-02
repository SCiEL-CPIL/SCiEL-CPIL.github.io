import numpy as np
import random
import time
import math
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, math.sqrt(3)/2])

def point_in_triangle(p, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def project_to_triangle(p):
    # Simple projection: if outside, move towards centroid
    # This is a heuristic. For SA, rejection is often better if we have a good move generator.
    # However, to keep points inside, we can use barycentric coordinates.
    # Let's stick to rejection in the move generation or clamping.
    # Clamping to the bounding box is not enough.
    # Let's use a "soft" constraint by penalizing out-of-bounds in the objective, 
    # or simply reject moves that go out of bounds.
    pass

def calc_min_area(points):
    # points: (N, 2)
    # Returns the minimum area of any triangle formed by 3 points
    n = len(points)
    min_a = float('inf')
    # Vectorized calculation is faster but N=11 is small. 
    # 165 triangles.
    # Let's use a simple loop for clarity and low overhead, or vectorize if needed.
    # Vectorized:
    # We can compute all pairwise differences, but area needs 3 points.
    # It's easier to just loop i, j, k.
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                area = 0.5 * abs(dx1*dy2 - dy1*dx2)
                if area < min_a:
                    min_a = area
    return min_a

def soft_min_area(points, tau):
    # For optimization
    n = len(points)
    areas = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                area = 0.5 * abs(dx1*dy2 - dy1*dx2)
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -tau * log(sum(exp(-A/tau)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    exps = np.exp(-(areas - max_a) / tau)
    sum_exps = np.sum(exps)
    soft_min = max_a - tau * np.log(sum_exps)
    return soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Random points inside triangle
    # Use barycentric coordinates for uniform sampling
    # u, v random, u+v < 1
    # P = (1-u-v)A + uB + vC
    def random_point():
        while True:
            u = random.random()
            v = random.random()
            if u + v < 1.0:
                return (1-u-v)*A + u*B + v*C

    best_points = None
    best_val = -1.0
    
    # Run multiple SA restarts
    # Budget: ~140s
    # Each SA run: ~20-40s
    # Let's do 3-4 restarts
    
    num_restarts = 4
    steps_per_restart = 20000
    
    for r in range(num_restarts):
        # Check time
        if time.time() > deadline - 20: # Leave 20s for polish and overhead
            break
            
        # Initialize
        points = np.array([random_point() for _ in range(N)])
        
        # Simulated Annealing
        T = 0.05  # Initial temperature (area scale)
        T_end = 1e-5
        step_temp = T
        current_val = calc_min_area(points)
        current_points = points.copy()
        
        # Precompute indices for speed? No, N is small.
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
            
            # Pick random point to move
            idx = random.randint(0, N-1)
            old_p = current_points[idx].copy()
            
            # Generate move
            # Gaussian noise
            noise = np.random.normal(0, step_temp, 2)
            new_p = old_p + noise
            
            # Check if inside triangle
            # Simple check: barycentric
            # P = (1-u-v)A + uB + vC
            # Solve for u,v
            # new_p = u*B + v*C (since A=0)
            # [1 0.5; 0 sqrt(3)/2] [u; v] = [x; y]
            # y = v * sqrt(3)/2 => v = 2y/sqrt(3)
            # x = u + 0.5v => u = x - 0.5v
            v_new = 2.0 * new_p[1] / math.sqrt(3)
            u_new = new_p[0] - 0.5 * v_new
            
            if u_new < 0 or v_new < 0 or u_new + v_new > 1.0:
                # Reject move or project?
                # Rejection is standard for SA.
                # But if we reject too much, we waste steps.
                # Let's try to project to the nearest point on the boundary?
                # Or just reject.
                continue
            
            # Update point
            current_points[idx] = new_p
            
            new_val = calc_min_area(current_points)
            
            # Acceptance
            delta = new_val - current_val
            if delta > 0 or random.random() < math.exp(delta / step_temp):
                current_val = new_val
                # Update best
                if current_val > best_val:
                    best_val = current_val
                    best_points = current_points.copy()
            else:
                # Revert
                current_points[idx] = old_p
            
            # Cool down
            step_temp *= 0.9995
            
            # Periodic reheat?
            if step % 5000 == 0 and step > 0:
                step_temp = T * 0.5
        
        # End of SA restart
        if time.time() > deadline - 15:
            break

    # Final Polish with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 5:
        # Use soft-min objective
        tau = 0.001
        def objective(x):
            pts = x.reshape(N, 2)
            # Check bounds? Nelder-Mead might go outside.
            # We need to penalize out-of-bounds.
            # Calculate barycentric coords for all points
            v_coords = 2.0 * pts[:, 1] / math.sqrt(3)
            u_coords = pts[:, 0] - 0.5 * v_coords
            # Penalty if u<0, v<0, u+v>1
            pen = 0.0
            for i in range(N):
                if u_coords[i] < 0: pen += (u_coords[i])**2
                if v_coords[i] < 0: pen += (v_coords[i])**2
                if u_coords[i] + v_coords[i] > 1.0: pen += (u_coords[i] + v_coords[i] - 1.0)**2
            if pen > 1e-6:
                return -soft_min_area(pts, tau) + pen * 100
            return -soft_min_area(pts, tau)
        
        x0 = best_points.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < -best_val:
                # Verify the result is valid
                final_pts = res.x.reshape(N, 2)
                # Check bounds
                v_coords = 2.0 * final_pts[:, 1] / math.sqrt(3)
                u_coords = final_pts[:, 0] - 0.5 * v_coords
                valid = True
                for i in range(N):
                    if u_coords[i] < -1e-9 or v_coords[i] < -1e-9 or u_coords[i] + v_coords[i] > 1.0 + 1e-9:
                        valid = False
                        break
                if valid:
                    final_val = calc_min_area(final_pts)
                    if final_val > best_val:
                        best_points = final_pts
                        best_val = final_val
        except Exception:
            pass

    # Ensure we return something
    if best_points is None:
        # Fallback to random
        best_points = np.array([random_point() for _ in range(N)])
        
    return best_points
