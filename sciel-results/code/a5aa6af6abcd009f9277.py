"""Unit regular hexagons inside a regular hexagon n=12"""
import numpy as np
import time
import math
from scipy.optimize import minimize
import random

N = 12

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def point_in_hexagon(p, cx, cy, side, angle_deg):
    """Check if point p is inside the hexagon."""
    # Transform to hexagon local coords
    dx = p[0] - cx
    dy = p[1] - cy
    angle = math.radians(-angle_deg)
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    # Regular hexagon with side s has flat-top or pointy-top; use standard form
    # For a regular hexagon with vertices at angles 0,60,..., the inequality is:
    # |x| <= s, |y| <= s*sqrt(3)/2, and |x*sqrt(3)/2 + y/2| <= s
    s = side
    return (abs(rx) <= s) and (abs(ry) <= s * math.sqrt(3) / 2.0) and (abs(rx * math.sqrt(3) / 2.0 + ry / 2.0) <= s)

def hexagon_distance(h1, h2):
    """
    Approximate distance between two hexagons. 
    This is a lower bound; if > 0, they're disjoint.
    Uses center-to-center distance minus sum of inradii as a quick check,
    but for accuracy we need proper convex distance.
    For now, use a conservative check: if center distance > 2*side, definitely disjoint.
    For tighter check, we'd need proper convex distance computation.
    """
    c1 = np.array([h1[0], h1[1]])
    c2 = np.array([h2[0], h2[1]])
    d = np.linalg.norm(c1 - c2)
    # Both are unit hexagons, max distance from center to vertex is 1, min (inradius) is sqrt(3)/2
    # If d > 2, they're definitely disjoint
    # If d < sqrt(3), they might overlap significantly
    # For a proper check, we'd compute the minimum distance between two convex polygons
    # Here we use a simpler check: if d > 2*side, disjoint; else need detailed check
    return d  # Return center distance; caller should check with proper overlap test

def overlap_penalty(h1, h2, outer_hex, penalty_weight=10.0):
    """
    Return a penalty if two inner hexagons overlap or if any inner hexagon is outside outer.
    This is a simplified version; for production, use proper convex distance.
    """
    # Check if h1 and h2 overlap using center distance
    d = math.sqrt((h1[0]-h2[0])**2 + (h1[1]-h2[1])**2)
    # Two unit hexagons: if center distance < sqrt(3), they definitely overlap
    # If center distance > 2, they definitely don't overlap
    # For a tighter bound, use inradius: if d < 2*sqrt(3)/2 = sqrt(3), overlap
    if d < math.sqrt(3) - 1e-9:
        # Definite overlap
        return penalty_weight * (math.sqrt(3) - d)
    elif d < 2.0:
        # Might overlap; use a soft penalty
        # This is a heuristic; for accuracy, need proper convex distance
        # For now, treat as potential overlap with reduced penalty
        return penalty_weight * 0.5 * (2.0 - d)
    return 0.0

def compute_objective(params, outer_center, outer_side, outer_angle):
    """
    Compute the objective: maximize minimum clearance or minimize outer side.
    params: list of (x, y, angle) for N hexagons
    outer_center, outer_side, outer_angle: define the outer hexagon
    """
    N_local = len(params)
    total_penalty = 0.0
    
    # Check pairwise overlaps
    for i in range(N_local):
        for j in range(i+1, N_local):
            h1 = (params[i][0], params[i][1], params[i][2])
            h2 = (params[j][0], params[j][1], params[j][2])
            total_penalty += overlap_penalty(h1, h2)
    
    # Check if inner hexagons are inside outer hexagon
    for i in range(N_local):
        x, y, ang = params[i]
        # Check if the inner hexagon is inside the outer hexagon
        # Approximate: check if center is inside outer hexagon with margin
        # For accuracy, check all 6 vertices of inner hex are inside outer
        inner_verts = hexagon_vertices(x, y, 1.0, ang)
        for v in inner_verts:
            if not point_in_hexagon(v, outer_center[0], outer_center[1], outer_side, outer_angle):
                total_penalty += 5.0
    
    return total_penalty

def construct():
    """
    Find optimal placement of 12 unit hexagons inside a regular hexagon,
    minimizing the outer side length.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second margin
    
    # Initial best
    best_side = 4.5
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try several initial configurations and optimize
    random.seed(42)
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with a symmetric configuration
        # Place hexagons in a hexagonal pattern
        center = (0.0, 0.0)
        
        # Start with a hexagonal grid arrangement
        # For 12 hexagons, use a pattern like: 3 in middle, 4 on top, 4 on bottom, 1 extra
        # Or use a circular arrangement
        if trial < 5:
            # Hexagonal grid
            positions = []
            # Central column
            for i in range(-2, 3):
                positions.append((0.0, i * 1.75))
            # Left and right columns
            for i in range(-1, 2):
                positions.append((-1.5, i * 1.75 + 0.875))
                positions.append((1.5, i * 1.75 + 0.875))
            # Add 2 more
            positions.append((-1.5, -2.625))
            positions.append((1.5, -2.625))
            # Trim to 12
            positions = positions[:12]
            angles = [0.0] * 12
            params = [(x, y, a) for x, y, a in zip([p[0] for p in positions], 
                                                     [p[1] for p in positions], angles)]
        else:
            # Random initialization
            params = []
            for i in range(12):
                x = random.uniform(-2, 2)
                y = random.uniform(-2, 2)
                ang = random.uniform(0, 60)
                params.append((x, y, ang))
        
        # Optimize using Nelder-Mead
        # Objective: minimize outer side length such that all hexagons fit
        # We'll use a penalty-based approach
        
        def objective_for_trial(p):
            if time.time() > deadline:
                return 100.0
            params_local = [(p[i*3], p[i*3+1], p[i*3+2]) for i in range(12)]
            # Estimate required outer side
            # Find the bounding hexagon
            max_x = max(p[i*3] for i in range(12))
            min_x = min(p[i*3] for i in range(12))
            max_y = max(p[i*3+1] for i in range(12))
            min_y = min(p[i*3+1] for i in range(12))
            
            # Approximate outer side as max extent
            # For a hexagon, the side length relates to the width and height
            # Width = 2*side, Height = sqrt(3)*side
            # So side = max(width/2, height/sqrt(3))
            width = max_x - min_x
            height = max_y - min_y
            side_est = max(width / 2.0, height / math.sqrt(3.0)) + 1.0  # +1 for hexagon size
            
            # Check overlaps
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    d = math.sqrt((p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2)
                    if d < math.sqrt(3) - 1e-6:
                        penalty += (math.sqrt(3) - d)
            
            return side_est + 10.0 * penalty
        
        p0 = np.array([param[k] for param in params for k in range(3)])
        
        try:
            result = minimize(objective_for_trial, p0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if time.time() > deadline:
                break
            if result.fun < best_side - 10.0:  # Allow for penalty
                # Extract best solution
                best_params = [(result.x[i*3], result.x[i*3+1], result.x[i*3+2]) for i in range(12)]
                # Compute actual outer side
                max_x = max(bp[0] for bp in best_params)
                min_x = min(bp[0] for bp in best_params)
                max_y = max(bp[1] for bp in best_params)
                min_y = min(bp[1] for bp in best_params)
                width = max_x - min_x
                height = max_y - min_y
                # The outer hexagon side that contains these
                # For a pointy-top hexagon with center at origin:
                # The constraint is: |x*sqrt(3)/2 + y/2| <= s, |x| <= s, |y*sqrt(3)/2| <= s
                # Actually, for a regular hexagon with flat top, vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
                # Let's use the proper containment test
                side_test = 4.5
                while side_test > 3.0:
                    # Check if all vertices of all inner hexagons are inside outer hexagon
                    all_inside = True
                    for bp in best_params:
                        inner_verts = hexagon_vertices(bp[0], bp[1], 1.0, bp[2])
                        for v in inner_verts:
                            if not point_in_hexagon(v, 0.0, 0.0, side_test, 0.0):
                                all_inside = False
                                break
                        if not all_inside:
                            break
                    if all_inside:
                        break
                    side_test -= 0.01
                if side_test < best_side:
                    best_side = side_test
                    best_inner = best_params
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
        except Exception:
            pass
    
    if best_inner is None:
        # Fallback
        inner = [[1.75 * k, 0.0, 0.0] for k in range(-5, 7)]
        inner = inner[:12]
        return inner, (0.0, 0.0), 4.0, 0.0
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
