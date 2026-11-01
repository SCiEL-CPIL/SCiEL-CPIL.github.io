import numpy as np
import math
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate lattice points
    points = []
    points.append((0, 0))
    
    ring1_coords = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q, r in ring1_coords:
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        points.append((x, y))
        
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring2.append((q, r))
                
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return (x, y)
        
    ring2_cart = [axial_to_cartesian(q, r) for q, r in ring2]
    ring2_cart.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    best_L = float('inf')
    best_config = None
    
    # Try subsets of 5 from ring 2
    # To save time, try contiguous blocks
    for start in range(12):
        if time.time() > deadline:
            break
            
        subset = []
        for i in range(5):
            subset.append(ring2_cart[(start + i) % 12])
            
        all_points = points + subset
        
        cx = sum(p[0] for p in all_points) / len(all_points)
        cy = sum(p[1] for p in all_points) / len(all_points)
        centered_points = [(p[0]-cx, p[1]-cy) for p in all_points]
        
        best_L_rot = float('inf')
        best_rot = 0
        
        # Sample rotation
        for theta_deg in np.linspace(0, 60, 31):
            if time.time() > deadline:
                break
                
            theta = math.radians(theta_deg)
            normals = []
            for k in range(6):
                angle = theta + k * math.pi / 3
                normals.append((math.cos(angle), math.sin(angle)))
            
            h_U_vals = []
            for nx, ny in normals:
                phi_n = math.atan2(ny, nx)
                max_dot = 0
                for v in range(6):
                    angle_v = v * math.pi / 3
                    dot = math.cos(phi_n - angle_v)
                    if dot > max_dot:
                        max_dot = dot
                h_U_vals.append(max_dot)
                
            min_L = 0
            for k in range(6):
                nx, ny = normals[k]
                dots = [p[0]*nx + p[1]*ny for p in centered_points]
                w = max(dots) - min(dots)
                req_w = w + 2 * h_U_vals[k]
                L_req = req_w / math.sqrt(3)
                if L_req > min_L:
                    min_L = L_req
                    
            if min_L < best_L_rot:
                best_L_rot = min_L
                best_rot = theta_deg
                
        if best_L_rot < best_L:
            best_L = best_L_rot
            best_config = (all_points, cx, cy, best_rot)
            
    if best_config is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        
    pts, cx, cy, rot = best_config
    
    inner = []
    for x, y in pts:
        inner.append([x - cx, y - cy, 0.0])
        
    return inner, (0.0, 0.0), best_L, rot

# EVOLVE-BLOCK-END
