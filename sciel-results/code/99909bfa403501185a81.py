"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if a point is inside a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        x_rot = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
        y_rot = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
        
        apothem = side * sqrt3 / 2.0
        
        for i in range(6):
            theta = math.radians(30.0 + i * 60.0)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = x_rot * nx + y_rot * ny
            if proj > apothem + 1e-9:
                return False
        return True

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two hexagons overlap. Returns True if they do."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Max distance between any two points in two unit hexagons is 2+2=4?
                # Actually, circumradius is 1. Max dist between centers for non-overlap is not simply 2.
                # But if dist > 2.0 + 2.0 = 4.0, definitely no overlap.
                # More tightly: If dist > 2.0 * (1 + 1) ? No.
                # The diameter of a hexagon is 2.
                # If center distance > 2 + 2 = 4, no overlap.
                # If center distance > 2.0, they might still overlap if aligned poorly?
                # No, the max extent from center is 1. So if dist > 2, they are disjoint?
                # Wait. The distance from center to vertex is 1.
                # If two hexagons have centers distance D apart.
                # The closest points can be as close as D - 2 (if aligned radially).
                # So if D > 2, they are definitely disjoint? 
                # No. Consider two hexagons side by side.
                # Distance between centers for touching sides: 
                # The apothem is sqrt(3)/2. Two hexagons touching sides have center distance 2*apothem = sqrt(3) ~ 1.732.
                # So D > 2 is a safe bound for "definitely no overlap", but D < 2 can still be disjoint.
                
                if dist > 2.0 + 1e-9:
                    continue
                
                # Detailed check
                verts_i = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                verts_j = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Check if any vertex of i is inside j
                for vx, vy in verts_i:
                    if point_in_hex(vx, vy, centers[j][0], centers[j][1], angles[j]):
                        return True
                # Check if any vertex of j is inside i
                for vx, vy in verts_j:
                    if point_in_hex(vx, vy, centers[i][0], centers[i][1], angles[i]):
                        return True
                        
        return False

    def required_L(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of outer hexagon to enclose all inner hexagons."""
        outer_angle_rad = math.radians(outer_angle)
        max_proj = 0.0
        
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for vx, vy in verts:
                x_rot = vx * math.cos(-outer_angle_rad) - vy * math.sin(-outer_angle_rad)
                y_rot = vx * math.sin(-outer_angle_rad) + vy * math.cos(-outer_angle_rad)
                
                for k in range(6):
                    theta = math.radians(30.0 + k * 60.0)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = x_rot * nx + y_rot * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / sqrt3

    def objective(params):
        """
        Objective function for optimization.
        params: [x1, y1, x2, y2, ..., x12, y12]
        Angles are fixed to 0 for this strategy (aligned with lattice).
        """
        centers = []
        for i in range(N):
            centers.append((params[2*i], params[2*i+1]))
        angles = [0.0] * N
        
        # Check overlap
        if check_overlap(centers, angles):
            # Penalty for overlap
            # Calculate a rough penalty based on how much they overlap?
            # Simple penalty: 100 + L
            L = required_L(centers, angles, 0.0)
            return L + 100.0
            
        L = required_L(centers, angles, 0.0)
        return L

    # Initial guess: Hexagonal grid arrangement
    # A cluster of 12 hexagons can be arranged as a central one + 5 in first ring + 6 in second?
    # Or 3x4 grid?
    # Let's try a compact cluster.
    # Hex grid coordinates:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1) ... 
    # Let's use axial coordinates and convert to cartesian.
    # Axial (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # We want 12 hexagons.
    # A "flower" of 7 (center + 6) is compact.
    # Add 5 more around it.
    
    coords_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), # Ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1) # Part of Ring 2
    ]
    
    init_centers = []
    for q, r in coords_axial:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        init_centers.append((x, y))
        
    init_params = []
    for cx, cy in init_centers:
        init_params.append(cx)
        init_params.append(cy)
        
    best_L = float('inf')
    best_params = init_params
    
    # Refine using Nelder-Mead
    # We fix angles to 0. This is a strong assumption but often optimal for hexagonal packing.
    # If the result is poor, we might need to allow rotation, but let's try this first.
    
    try:
        res = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res.fun < best_L:
            best_L = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    # The Nelder-Mead can get stuck in local minima where hexagons are "jammed" suboptimally.
    # Random restarts with small perturbations can help escape.
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and restarts < max_restarts:
        # Perturb the current best solution
        perturb_scale = 0.5
        perturbed_params = best_params + np.random.normal(0, perturb_scale, size=len(best_params))
        
        try:
            res2 = minimize(
                objective, 
                perturbed_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res2.fun < best_L:
                best_L = res2.fun
                best_params = res2.x
        except Exception:
            pass
        restarts += 1
        
    # Extract final solution
    centers = []
    for i in range(N):
        centers.append((best_params[2*i], best_params[2*i+1]))
    angles = [0.0] * N
    
    # Final validity check
    if check_overlap(centers, angles):
        # Fallback to a known valid, though large, configuration
        # Line up in a row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    final_L = required_L(centers, angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
