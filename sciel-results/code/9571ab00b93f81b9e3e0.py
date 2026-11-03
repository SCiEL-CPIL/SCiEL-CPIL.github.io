import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (pointy-top at 0 deg? No, standard math angle 0 is right).
    Let's use vertices at angles 0, 60, 120, 180, 240, 300 relative to the rotation angle.
    This is a 'pointy-top' if angle=0? No, vertex at 0 is right. 
    Standard flat-top has vertices at 0, 60... wait.
    If vertex is at 0, it's pointy-right.
    If we want flat-top, vertices are at 30, 90, 150...
    Let's stick to vertices at 0, 60, ... for simplicity, and let rotation handle orientation.
    """
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _sat_overlap_depth(v1, v2, tol=1e-9):
    """
    Returns the maximum overlap depth between two convex polygons.
    Returns 0 if they do not overlap.
    """
    min_overlap = float('inf')
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            # Project vertices
            p1 = v1 @ normal
            p2 = v2 @ normal
            
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            
            # Check for separation
            if max1 < min2 + tol or max2 < min1 + tol:
                return 0.0
            
            # Overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
    return min_overlap if min_overlap < float('inf') else 0.0

def _min_outer_side(verts_list, outer_angle=0.0):
    """
    Calculate the minimum side length S of an outer hexagon (centered at 0,0, rotated by outer_angle)
    that contains all hexagons in verts_list.
    """
    a = math.radians(outer_angle)
    # Normals of the outer hexagon
    # The outer hexagon has edges perpendicular to angles 30, 90, 150, 210, 270, 330 + outer_angle
    # Wait, vertices of outer hex are at 0, 60... + outer_angle.
    # Edge normals are at 30, 90... + outer_angle.
    # Distance from center to edge is inradius = S * sqrt(3) / 2.
    # So S = 2 * inradius / sqrt(3).
    
    max_proj = 0.0
    for i in range(6):
        theta = a + math.pi / 6.0 + i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        # Project all vertices of all inner hexagons onto this normal
        for verts in verts_list:
            proj = verts @ np.array([nx, ny])
            if proj.max() > max_proj:
                max_proj = proj.max()
                
    # max_proj is the maximum distance from origin to any vertex along the normal direction.
    # This distance must be <= inradius = S * sqrt(3) / 2
    # So S >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def _objective(params):
    """
    Objective function for the optimizer.
    params: array of size 36 (12 * 3). [x0, y0, theta0, x1, y1, theta1, ...]
    Returns: S_outer + penalty * sum(overlap^2)
    """
    # Unpack
    positions = np.array(params).reshape(N, 3)
    # Angles are in radians
    xs = positions[:, 0]
    ys = positions[:, 1]
    thetas = positions[:, 2]
    
    # Generate vertices
    verts_list = []
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], math.degrees(thetas[i]), 1.0)
        verts_list.append(v)
        
    # Calculate minimum outer side
    S = _min_outer_side(verts_list, 0.0)
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                penalty += depth * depth
                
    # We want to minimize S, but strongly penalize overlap.
    # If penalty is large, S is irrelevant.
    # Use a large multiplier for penalty.
    return S + 100.0 * penalty

def _generate_initial_guesses():
    """
    Generate several initial configurations.
    """
    guesses = []
    
    # Guess 1: Honeycomb 1-6-5
    # Center
    guess1 = np.zeros((N, 3))
    # Ring 1: 6 hexagons
    dist1 = SQRT3
    for i in range(6):
        angle = i * math.pi / 3.0
        guess1[i+1, 0] = dist1 * math.cos(angle)
        guess1[i+1, 1] = dist1 * math.sin(angle)
        guess1[i+1, 2] = 0.0
    # Ring 2: 5 hexagons. Pick 5 from 12 positions.
    # Positions in Ring 2 are at distance 2*sqrt(3) and angles 0, 30, 60... (offset by 30 deg from Ring 1 vertices? No, centers are further out).
    # In a hex grid, the 2nd ring centers are at distance 2*sqrt(3).
    # Angles are 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.
    # Let's pick 5 that are symmetric-ish or just the first 5.
    # To minimize bounding box, maybe pick ones that are "inwards" or balanced.
    # Let's pick angles: 0, 60, 120, 180, 240 (every other one? No, that's 5).
    # Actually, let's just pick 5 consecutive ones to form a cluster, or spread them.
    # Spreading them might increase S. Clustering might allow smaller S if the rest fill the center.
    # Let's try 5 spread out: 0, 72, 144, 216, 288? No, hex symmetry is 60 deg.
    # Let's pick 0, 60, 120, 180, 240.
    dist2 = 2.0 * SQRT3
    for i in range(5):
        angle = i * math.pi / 3.0
        guess1[i+7, 0] = dist2 * math.cos(angle)
        guess1[i+7, 1] = dist2 * math.sin(angle)
        guess1[i+7, 2] = 0.0
    guesses.append(guess1)
    
    # Guess 2: 3x4 Grid (Rectangular packing)
    # Hexagons in a row: dx = 1.5, dy = 0
    # Next row: dx = 1.5, dy = sqrt(3)/2 * 2? No.
    # For flat-top (vertices at 0, 60...), width=2, height=sqrt(3).
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3) * 1.5? No.
    # Let's use the lattice vectors.
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (1.5, -sqrt(3)/2)
    # This creates a hex grid.
    # We can just place them in a grid using these vectors.
    guess2 = np.zeros((N, 3))
    idx = 0
    for i in range(4): # 4 columns
        for j in range(3): # 3 rows
            x = i * 1.5 + (j % 2) * 0.75
            y = j * SQRT3 / 2.0
            # Center the grid
            guess2[idx, 0] = x
            guess2[idx, 1] = y
            guess2[idx, 2] = 0.0
            idx += 1
    # Center the whole grid
    cx = np.mean(guess2[:, 0])
    cy = np.mean(guess2[:, 1])
    guess2[:, 0] -= cx
    guess2[:, 1] -= cy
    guesses.append(guess2)
    
    # Guess 3: Random
    guess3 = np.random.rand(N, 3) * 10 - 5
    guess3[:, 2] = np.random.rand(N) * 2 * math.pi
    guesses.append(guess3)
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_S = float('inf')
    best_params = None
    
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B with numerical gradients
            # Bounds: positions can be anywhere, but let's restrict to reasonable range
            bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N
            
            res = minimize(
                _objective, 
                guess.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if res.success or res.fun < best_S:
                # Check validity: is overlap actually 0?
                params = res.x.reshape(N, 3)
                verts_list = []
                for i in range(N):
                    v = _hex_vertices(params[i, 0], params[i, 1], math.degrees(params[i, 2]), 1.0)
                    verts_list.append(v)
                
                overlap_exists = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if _sat_overlap_depth(verts_list[i], verts_list[j]) > 1e-6:
                            overlap_exists = True
                            break
                    if overlap_exists:
                        break
                
                if not overlap_exists:
                    S = _min_outer_side(verts_list, 0.0)
                    if S < best_S:
                        best_S = S
                        best_params = params
                        
        except Exception as e:
            pass

    # If no valid solution found, return a fallback
    if best_params is None:
        # Fallback: simple grid, even if it overlaps or is large
        inner = []
        for i in range(N):
            inner.append([i * 2.0, 0.0, 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0

    # Format output
    inner = []
    for i in range(N):
        x, y, theta_rad = best_params[i]
        theta_deg = math.degrees(theta_rad) % 360.0
        inner.append([float(x), float(y), float(theta_deg)])
        
    # Outer center is (0,0), side is best_S, angle is 0
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
