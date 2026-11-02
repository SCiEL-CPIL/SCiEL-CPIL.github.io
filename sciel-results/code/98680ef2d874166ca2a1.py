import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        # We compute the support function of the union of hexagons.
        # The outer hexagon (centered at origin, angle 0) has normals at 30, 90, 150, 210, 270, 330.
        # For a hexagon of side S centered at origin, the support in direction n is S * sqrt(3) / 2.
        # So we need max_i (support_i(n)) <= S * sqrt(3) / 2.
        # S >= 2/sqrt(3) * max_i (support_i(n)) for all 6 normals n.
        
        sqrt3 = np.sqrt(3.0)
        max_proj = 0.0
        
        # Normals for outer hexagon (flat top/bottom, vertices at 0, 60...)
        # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
        # n_1 = (cos(90), sin(90)) = (0, 1)
        # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos(210), sin(210)) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos(270), sin(270)) = (0, -1)
        # n_5 = (cos(330), sin(330)) = (sqrt(3)/2, -1/2)
        
        normals = np.array([
            [sqrt3/2, 0.5],
            [0.0, 1.0],
            [-sqrt3/2, 0.5],
            [-sqrt3/2, -0.5],
            [0.0, -1.0],
            [sqrt3/2, -0.5]
        ])
        
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            # Support of this hexagon in each normal direction
            projs = verts @ normals.T
            max_support = np.max(projs)
            if max_support > max_proj:
                max_proj = max_support
                
        return (2.0 / sqrt3) * max_proj

    def check_overlap_penalty(positions, angles, tol=1e-3):
        """
        Compute a penalty term for overlapping hexagons.
        Uses a simple distance-based heuristic for speed, or exact SAT if needed.
        For Nelder-Mead, a smooth penalty is better.
        We approximate the minimum distance between hexagons.
        If centers are closer than 2*sqrt(3) (approx 3.46) for aligned hexagons, they might overlap.
        Actually, the inradius is sqrt(3)/2 ~ 0.866. 
        Distance between centers must be at least 2 * inradius = sqrt(3) ~ 1.732 for parallel sides to touch.
        But with rotation, it's more complex.
        
        Let's use a simple pairwise distance check. 
        If dist < sqrt(3) - tol, they definitely overlap (for any rotation? No, if one is rotated 30 deg, 
        the "width" in some direction is 2, inradius is sqrt(3)/2. 
        The minimum distance between centers for non-overlap is at least 2 * inradius = sqrt(3).
        So if dist < sqrt(3), they overlap.
        If dist >= 2 * circumradius = 2, they don't overlap.
        Between sqrt(3) and 2, it depends on rotation.
        
        For a robust penalty, let's compute the exact minimum distance or use SAT.
        Given the time limit, let's use a simplified SAT check that returns a "penetration depth" estimate.
        """
        penalty = 0.0
        sqrt3 = np.sqrt(3.0)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # If far apart, no penalty
                if dist >= 2.0:
                    continue
                
                # If very close, high penalty
                # Theoretical min distance for non-overlap is sqrt(3) ~ 1.732
                if dist < sqrt3:
                    # Simple linear penalty for violation
                    penalty += (sqrt3 - dist) ** 2
                else:
                    # In the intermediate zone, we need to check rotation.
                    # To save time, we assume a small penalty if they are close, 
                    # but let's try a quick SAT check for the "worst case" or just skip 
                    # and rely on the optimizer to push them apart if they are actually overlapping.
                    # Actually, if dist is between sqrt(3) and 2, they *might* overlap.
                    # Let's do a fast SAT check.
                    v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                    v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                    
                    # SAT check
                    overlap = True
                    # Check normals of v1
                    for k in range(6):
                        edge = v1[(k+1)%6] - v1[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10: continue
                        normal /= norm
                        
                        proj1 = np.dot(v1, normal)
                        proj2 = np.dot(v2, normal)
                        
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        if max1 + tol < min2 or max2 + tol < min1:
                            overlap = False
                            break
                    
                    if overlap:
                        # Check normals of v2
                        for k in range(6):
                            edge = v2[(k+1)%6] - v2[k]
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm < 1e-10: continue
                            normal /= norm
                            
                            proj1 = np.dot(v1, normal)
                            proj2 = np.dot(v2, normal)
                            
                            min1, max1 = np.min(proj1), np.max(proj1)
                            min2, max2 = np.min(proj2), np.max(proj2)
                            
                            if max1 + tol < min2 or max2 + tol < min1:
                                overlap = False
                                break
                        
                        if overlap:
                            # They overlap. Estimate penetration depth.
                            # A crude estimate is based on how much the projections overlap on the worst axis.
                            # For now, just add a fixed penalty to push them apart.
                            penalty += 1.0
                            
        return penalty

    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        """
        positions = params[0::3].reshape(-1, 2)
        angles = params[1::3]
        
        side = required_outer_side(positions, angles)
        overlap_pen = check_overlap_penalty(positions, angles)
        
        # Penalty weight needs to be strong enough to prevent overlap
        # side is around 4-5. overlap_pen can be large.
        return side + 100.0 * overlap_pen

    # Initial guess: Hexagonal packing
    # 1 center, 6 ring 1, 5 ring 2
    r1 = math.sqrt(3) # distance for touching
    r2 = 2 * math.sqrt(3)
    
    x0 = []
    # Center
    x0.extend([0.0, 0.0, 0.0])
    # Ring 1
    for i in range(6):
        theta = np.radians(60 * i)
        x0.extend([r1 * np.cos(theta), r1 * np.sin(theta), 0.0])
    # Ring 2
    # Place 5 hexagons in the gaps of ring 1? Or just spaced out.
    # Let's space them by 72 degrees starting from 30 deg offset to align with gaps
    offset = 30.0
    for i in range(5):
        theta = np.radians(offset + 72 * i)
        x0.extend([r2 * np.cos(theta), r2 * np.sin(theta), 0.0])
    
    # Run Nelder-Mead
    # We need to be careful with the number of iterations.
    # 36 dimensions is a lot.
    
    # To improve success rate, we can try multiple restarts or a better initial guess.
    # Let's try a few different initial configurations.
    
    best_side = float('inf')
    best_params = None
    
    # Define a few initial configurations
    configs = []
    
    # Config 1: Standard hexagonal
    configs.append(x0)
    
    # Config 2: Rotated ring 1
    x0_rot = []
    x0_rot.extend([0.0, 0.0, 0.0])
    for i in range(6):
        theta = np.radians(60 * i + 30)
        x0_rot.extend([r1 * np.cos(theta), r1 * np.sin(theta), 30.0]) # Rotate hexagons too
    offset = 0.0
    for i in range(5):
        theta = np.radians(offset + 72 * i)
        x0_rot.extend([r2 * np.cos(theta), r2 * np.sin(theta), 0.0])
    configs.append(x0_rot)
    
    # Config 3: Linear arrangement (fallback)
    x0_lin = []
    for i in range(N):
        x0_lin.extend([i * 2.0, 0.0, 0.0])
    configs.append(x0_lin)

    for config in configs:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(objective, config, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < best_side:
                # Verify validity
                positions = res.x[0::3].reshape(-1, 2)
                angles = res.x[1::3]
                
                # Strict validity check
                valid = True
                sqrt3 = math.sqrt(3.0)
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                        v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                        
                        # SAT
                        overlap = True
                        for k in range(6):
                            edge = v1[(k+1)%6] - v1[k]
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm < 1e-10: continue
                            normal /= norm
                            proj1 = np.dot(v1, normal)
                            proj2 = np.dot(v2, normal)
                            if np.max(proj1) + 1e-6 < np.min(proj2) or np.max(proj2) + 1e-6 < np.min(proj1):
                                overlap = False
                                break
                        if overlap:
                            for k in range(6):
                                edge = v2[(k+1)%6] - v2[k]
                                normal = np.array([-edge[1], edge[0]])
                                norm = np.linalg.norm(normal)
                                if norm < 1e-10: continue
                                normal /= norm
                                proj1 = np.dot(v1, normal)
                                proj2 = np.dot(v2, normal)
                                if np.max(proj1) + 1e-6 < np.min(proj2) or np.max(proj2) + 1e-6 < np.min(proj1):
                                    overlap = False
                                    break
                            if overlap:
                                valid = False
                                break
                    if not valid:
                        break
                        
                if valid:
                    side = required_outer_side(positions, angles)
                    if side < best_side:
                        best_side = side
                        best_params = res.x.copy()
                        
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to the initial config if optimization failed or found nothing valid
        best_params = x0
        positions = best_params[0::3].reshape(-1, 2)
        angles = best_params[1::3]
        best_side = required_outer_side(positions, angles)

    # Final extraction
    positions = best_params[0::3].reshape(-1, 2)
    angles = best_params[1::3]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
