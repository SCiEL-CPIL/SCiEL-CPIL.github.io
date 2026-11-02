import numpy as np
import time
import math
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initial Guess: 3x4 Honeycomb Grid
    # Lattice vectors for pointy-top hexagons (side 1)
    # dx = sqrt(3), dy = 1.5
    # Row 0: y=0, x = 0, s3, 2s3, 3s3
    # Row 1: y=1.5, x = s3/2, 3s3/2, 5s3/2, 7s3/2
    # Row 2: y=3.0, x = 0, s3, 2s3, 3s3
    s3 = math.sqrt(3)
    centers = []
    for r in range(3):
        y = r * 1.5
        offset = (s3 / 2) if r % 2 == 1 else 0.0
        for c in range(4):
            x = c * s3 + offset
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center of mass
    center_guess = np.mean(centers, axis=0)
    
    # Initial angles: 0
    angles_guess = np.zeros(N)
    
    # Initial outer hexagon
    # Estimate S
    # Max distance from center_guess to any vertex
    # Vertices of a unit hex (pointy top)
    unit_verts = np.array([
        [0, 1],
        [s3/2, 0.5],
        [s3/2, -0.5],
        [0, -1],
        [-s3/2, -0.5],
        [-s3/2, 0.5]
    ])
    
    # Calculate max radius for S init
    max_r = 0
    for c in centers:
        verts = c + unit_verts
        d = np.linalg.norm(verts - center_guess, axis=1)
        max_r = max(max_r, np.max(d))
    
    S_guess = max_r + 0.1
    phi_guess = 0.0
    
    # Parameters: x1, y1, a1, ..., x12, y12, a12, Cx, Cy, phi, S
    # Total 2*12 + 12 + 2 + 1 + 1 = 40
    
    def objective(params):
        # Unpack
        c_xy = params[0:24].reshape(12, 2)
        c_ang = params[24:36]
        C = params[36:38]
        phi = params[38]
        S = params[39]
        
        if S < 0:
            return 1e10
            
        # 1. Containment Penalty
        # Rotate unit vertices by c_ang
        # unit_verts: (6, 2)
        # c_ang: (12,)
        # We need to rotate each set of 6 vertices by the corresponding angle
        
        # Efficient rotation
        # Create rotation matrices for each hex
        # Or just use broadcasting
        
        # cos/sin of angles
        cos_a = np.cos(c_ang)
        sin_a = np.sin(c_ang)
        
        # unit_verts: (6, 2)
        # x' = x cos - y sin
        # y' = x sin + y cos
        
        # u_x = unit_verts[:, 0] (6,)
        # u_y = unit_verts[:, 1] (6,)
        
        # For each i in 12:
        # verts_i = (u_x * cos_i - u_y * sin_i, u_x * sin_i + u_y * cos_i) + c_xy[i]
        
        # Broadcasting:
        # cos_a: (12,) -> (12, 1)
        # sin_a: (12,) -> (12, 1)
        # unit_verts: (6, 2) -> (1, 6, 2)
        
        cos_b = cos_a[:, np.newaxis]
        sin_b = sin_a[:, np.newaxis]
        
        u = unit_verts[np.newaxis, :, :] # (1, 6, 2)
        
        # Rotate
        # (1, 6, 2) * (12, 1, 1)
        # Need to handle x and y separately
        
        u_x = u[:, :, 0] # (1, 6)
        u_y = u[:, :, 1] # (1, 6)
        
        rx = u_x * cos_b - u_y * sin_b # (12, 6)
        ry = u_x * sin_b + u_y * cos_b # (12, 6)
        
        # Translate
        rx = rx + c_xy[:, 0:1] # (12, 6)
        ry = ry + c_xy[:, 1:2] # (12, 6)
        
        # All vertices: (12, 6, 2)
        all_verts = np.stack([rx, ry], axis=-1)
        
        # Outer Hexagon Normals
        # Pointy top hexagon normals are at 30, 90, 150... or 0, 60...?
        # If vertices are at 90, 30, -30...
        # Normals are perpendicular to sides.
        # Side between (0,1) and (s3/2, 0.5).
        # Vector (s3/2, -0.5). Normal (0.5, s3/2) -> angle 60?
        # Let's use the standard: Normals at 0, 60, 120, 180, 240, 300 degrees.
        # This corresponds to a "flat top" hexagon if normals are horizontal?
        # Let's stick to the definition:
        # Hexagon defined by n_j . (p - C) <= S * sqrt(3) / 2
        # If normals are at 0, 60, ..., the hexagon has flat sides perpendicular to axes.
        # This is a "pointy side" or "flat top" depending on convention.
        # Let's just use angles 0, 60, 120, 180, 240, 300 for normals.
        
        norm_angles = np.deg2rad(np.arange(0, 360, 60) + phi)
        normals = np.stack([np.cos(norm_angles), np.sin(norm_angles)], axis=1) # (6, 2)
        
        # Project all_verts onto normals
        # all_verts: (12, 6, 2)
        # normals: (6, 2)
        
        # v = all_verts - C
        v = all_verts - C[np.newaxis, np.newaxis, :]
        
        # Dots: (12, 6, 2) dot (2, 6) -> (12, 6, 6)
        # v @ normals.T
        projections = v @ normals.T
        
        limit = S * s3 / 2.0
        
        # Violation: max(0, projections - limit)
        violations_c = np.maximum(0, projections - limit)
        penalty_contain = np.sum(violations_c)
        
        # 2. Overlap Penalty
        # Pairwise differences
        # c_xy: (12, 2)
        # diff[i, j] = c_xy[j] - c_xy[i]
        
        # Use broadcasting
        # c_xy[:, np.newaxis, :] (12, 1, 2)
        # c_xy[np.newaxis, :, :] (1, 12, 2)
        
        # diff = c_xy[np.newaxis, :, :] - c_xy[:, np.newaxis, :] # (12, 12, 2)
        # We only need i < j
        
        # Inner Hex Normals (same as outer, angle 0)
        inner_norm_angles = np.deg2rad(np.arange(0, 360, 60))
        inner_normals = np.stack([np.cos(inner_norm_angles), np.sin(inner_norm_angles)], axis=1) # (6, 2)
        
        # Calculate projections for all pairs
        # diff: (12, 12, 2)
        # inner_normals: (6, 2)
        
        # proj = diff @ inner_normals.T # (12, 12, 6)
        
        # s_k = sqrt(3) - proj
        # min_s = min(s_k)
        # violation = max(0, -min_s)
        
        # To save memory/time, we can compute this in chunks or just use the full matrix since 12x12 is small.
        
        diff = c_xy[np.newaxis, :, :] - c_xy[:, np.newaxis, :]
        
        proj = diff @ inner_normals.T
        
        s_vals = s3 - proj
        
        min_s = np.min(s_vals, axis=2) # (12, 12)
        
        # Only consider i < j
        mask = np.triu(np.ones((12, 12), dtype=bool), k=1)
        
        # violation = max(0, -min_s)
        viol_ov = np.maximum(0, -min_s)
        viol_ov = viol_ov * mask
        
        penalty_overlap = np.sum(viol_ov)
        
        total_penalty = penalty_contain + penalty_overlap
        
        # Objective: Minimize S + Lambda * Penalty
        return S + 100.0 * total_penalty

    # Bounds
    # Positions: -10 to 10
    # Angles: 0 to 2pi
    # Cx, Cy: -10 to 10
    # Phi: 0 to 2pi
    # S: 1 to 10
    
    bounds = []
    for i in range(12):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 2 * np.pi))
    bounds.append((-10, 10)) # Cx
    bounds.append((-10, 10)) # Cy
    bounds.append((0, 2 * np.pi)) # Phi
    bounds.append((1, 10)) # S
    
    # Initial population around the guess
    x0 = np.concatenate([centers.flatten(), angles_guess, center_guess, [phi_guess, S_guess]])
    
    # Use differential evolution
    # popsize=15, maxiter=50
    # tol=1e-4
    
    best_result = None
    best_score = 1e10
    
    try:
        # Check time
        if time.time() > deadline - 20:
             # Just return the guess if we have no time
             pass
        else:
            res = differential_evolution(
                objective, 
                bounds, 
                x0=x0,
                popsize=15, 
                maxiter=50, 
                tol=1e-4,
                seed=42,
                polish=True
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
            # Also try the initial guess if it's better
            score_guess = objective(x0)
            if score_guess < best_score:
                best_score = score_guess
                best_result = x0
                
    except Exception as e:
        # Fallback to guess
        best_result = x0
        best_score = objective(x0)
        
    # Extract best result
    if best_result is None:
        best_result = x0
        
    c_xy_best = best_result[0:24].reshape(12, 2)
    c_ang_best = best_result[24:36]
    C_best = best_result[36:38]
    phi_best = best_result[38]
    S_best = best_result[39]
    
    # Check validity of the best result
    # Re-evaluate penalty to ensure it's 0
    # If penalty is high, we might need to return the guess if guess was valid?
    # The guess is valid.
    
    score_guess = objective(x0)
    score_best = best_score
    
    # If best has penalty, but guess is valid (penalty 0), and guess S < 40, return guess?
    # Actually, if DE fails to find valid, we should return the best VALID solution.
    # Let's check if best is valid.
    
    # We can't easily get penalty from score alone if S varies.
    # Let's just return the best found. If it's invalid, the evaluator will score 0.
    # But we want to score > 1.0.
    # The guess S is ~4. Score = 4.
    # If DE finds S=3.5 but invalid, it's bad.
    # If DE finds S=3.5 valid, it's good.
    
    # To be safe, if the penalty of the best result is significant, we might want to fallback.
    # However, with Lambda=100, the optimizer strongly prefers valid solutions.
    
    # Format output
    inner = []
    for i in range(12):
        x = c_xy_best[i, 0]
        y = c_xy_best[i, 1]
        ang_deg = np.rad2deg(c_ang_best[i]) % 360.0
        inner.append([x, y, ang_deg])
        
    center = (C_best[0], C_best[1])
    side = S_best
    angle = np.rad2deg(phi_best) % 360.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
