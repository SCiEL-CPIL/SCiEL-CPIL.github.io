"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines multi-topology initialization with local Nelder-Mead optimization.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0  # Safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    # A and B are numpy arrays of shape (6, 2)
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            projA = A[:, 0] * nx + A[:, 1] * ny
            minA = np.min(projA)
            maxA = np.max(projA)
            # project B
            projB = B[:, 0] * nx + B[:, 1] * ny
            minB = np.min(projB)
            maxB = np.max(projB)
            if maxA < minB - tol or maxB < minA - tol:
                return False  # separating axis found
    return True

def check_validity(inner_params, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping.
    inner_params: flat array [x1, y1, a1, ..., x12, y12, a12]"""
    positions = inner_params[:2*N].reshape(N, 2)
    angles = inner_params[2*N:]
    
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        polys.append(p)
    
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner_params, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side s, apothem = s * sqrt(3)/2.
    A point is inside iff for each of the 6 outward normals n_k, (p - c)·n_k <= apothem.
    So S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
    positions = inner_params[:2*N].reshape(N, 2)
    angles = inner_params[2*N:]
    
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
        
    max_proj = 0.0
    for i in range(N):
        p = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        # Vectorized projection calculation
        # p is (6,2), normals is list of 6 tuples
        # We need max over all vertices and all normals
        # proj = (p - c) dot n
        # Instead of looping vertices, we can just loop normals and use vector ops on p
        for (nx, ny) in normals:
            # (p - c) dot n
            # p is array, c is scalar
            proj = (p[:, 0] - cx) * nx + (p[:, 1] - cy) * ny
            local_max = np.max(proj)
            if local_max > max_proj:
                max_proj = local_max
                
    return (2.0 / math.sqrt(3.0)) * max_proj

def objective(params, outer_rot):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., x12, y12, a12]
    Returns side length + penalty if invalid.
    """
    # Unroll positions and angles
    # We don't need to check validity every single time for speed, 
    # but the penalty needs to be smooth-ish or at least strongly discouraging.
    # For Nelder-Mead, a hard penalty works if the initial guess is close to valid.
    
    side = compute_min_outer_side(params, (0.0, 0.0), outer_rot)
    
    # Check validity
    if not check_validity(params, (0.0, 0.0), side, outer_rot):
        # Calculate overlap depth for better penalty? 
        # Simple penalty: large constant + side
        # A better penalty: sum of overlap depths
        # For now, use a large penalty to steer away
        return side + 50.0
        
    return side

def build_ring12(scale, inner_rot):
    """Ring of 12 hexagons around center."""
    params = np.zeros(3*N)
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        params[3*k] = x
        params[3*k+1] = y
        params[3*k+2] = inner_rot
    return params

def build_hex6_plus6(scale1, scale2, rot1, rot2):
    """Two concentric rings of 6, no center."""
    params = np.zeros(3*N)
    idx = 0
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        params[3*idx] = x
        params[3*idx+1] = y
        params[3*idx+2] = rot1
        idx += 1
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        params[3*idx] = x
        params[3*idx+1] = y
        params[3*idx+2] = rot2
        idx += 1
    return params

def build_rect3x4(scale_x, scale_y, rot):
    """3x4 rectangular grid."""
    params = np.zeros(3*N)
    idx = 0
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            params[3*idx] = x
            params[3*idx+1] = y
            params[3*idx+2] = rot
            idx += 1
    return params

def optimize_initial_guess(x0, outer_rot, time_budget):
    """
    Run Nelder-Mead on a specific initial guess.
    Returns best params and best side.
    """
    start_time = time.time()
    
    # We want to minimize side length.
    # Nelder-Mead is derivative-free, good for this.
    
    # Options:
    # maxiter: limit steps
    # xatol, fatol: convergence
    # We don't have a strict time limit per call easily, but we can check in the callback?
    # SciPy minimize doesn't support easy time-based stopping in all versions/methods without callbacks.
    # However, Nelder-Mead is usually fast enough. 
    # To be safe, we can wrap the objective to check time? No, that's messy.
    # We'll rely on maxiter being reasonable.
    
    # Let's try to make it robust.
    try:
        res = minimize(objective, x0, args=(outer_rot,), method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        # Check if it finished within budget roughly?
        # If it took too long, we might want to discard? 
        # But we are inside a loop with a global deadline check before starting each guess.
        
        # Verify result
        test_side = compute_min_outer_side(res.x, (0.0, 0.0), outer_rot)
        if check_validity(res.x, (0.0, 0.0), test_side, outer_rot):
            return res.x, test_side
        else:
            # If invalid, try to repair? Or just return invalid (penalized)
            # The objective returns side + 50 if invalid. 
            # If the best found is invalid, it might be better than a valid one with large side?
            # No, we want a VALID solution.
            # If res.x is invalid, we should probably not return it as a "best valid" candidate.
            # But maybe we can return the best VALID point seen during optimization?
            # Nelder-Mead doesn't easily expose history of valid points.
            # We can just check the final result. If invalid, we return None or a large penalty.
            # Let's return None to indicate failure for this guess.
            return None, float('inf')
            
    except Exception as e:
        # print(f"Optimization failed: {e}")
        return None, float('inf')

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    
    # Define initial guesses
    # We will try a few different topologies and outer rotations
    
    outer_rotations = [0.0, 30.0] # Hexagon has 60 deg symmetry, 30 is offset
    
    guesses = []
    
    # 1. Ring 12
    for orot in outer_rotations:
        # Scale needs to be such that they don't overlap.
        # Distance between centers in ring: 2 * scale * sin(15 deg)
        # Min distance for hexagons is sqrt(3) ~ 1.732 (if aligned) or slightly less if rotated?
        # Actually, min distance between centers of two touching unit hexagons is sqrt(3) if they share an edge.
        # So 2 * scale * sin(15) >= sqrt(3) => scale >= sqrt(3) / (2 * 0.2588) ~ 1.732 / 0.5176 ~ 3.34
        # Let's try scales around 3.0 - 3.5
        for scale in [3.0, 3.3, 3.5]:
            for inner_rot in [0.0, 15.0]:
                x0 = build_ring12(scale, inner_rot)
                guesses.append((x0, orot))
                
    # 2. Hex 6+6
    for orot in outer_rotations:
        for s1 in [1.8, 2.0]:
            for s2 in [3.0, 3.2]:
                for r1 in [0.0]:
                    for r2 in [0.0, 30.0]:
                        x0 = build_hex6_plus6(s1, s2, r1, r2)
                        guesses.append((x0, orot))

    # 3. Rect 3x4
    for orot in outer_rotations:
        for sx in [1.8, 2.0]:
            for sy in [1.8, 2.0]:
                for rot in [0.0, 30.0]:
                    x0 = build_rect3x4(sx, sy, rot)
                    guesses.append((x0, orot))
                    
    # 4. Random
    np.random.seed(42)
    for _ in range(5):
        p = np.random.randn(N, 2) * 3.0
        a = np.random.rand(N) * 360.0
        x0 = np.concatenate([p.flatten(), a])
        guesses.append((x0, 0.0))
        
    # Sort guesses by initial objective? 
    # This might be expensive if we compute objective for all.
    # Let's just iterate.
    
    for i, (x0, orot) in enumerate(guesses):
        if time.time() > DEADLINE - 10:
            break
            
        # Time budget per guess: remaining time / remaining guesses?
        # Or just a fixed small budget?
        # Nelder-Mead with 36 dims can be slow.
        # Let's give it a chance.
        
        params, side = optimize_initial_guess(x0, orot, 10.0)
        
        if params is not None and side < best_side:
            best_side = side
            best_params = params.copy()
            
    # If no valid solution found, we have a problem.
    # But the initial guesses (like ring 12 with large scale) should be valid.
    # Let's ensure we have a fallback.
    if best_params is None:
        # Try a very large ring
        x0 = build_ring12(5.0, 0.0)
        side = compute_min_outer_side(x0, (0.0, 0.0), 0.0)
        if check_validity(x0, (0.0, 0.0), side, 0.0):
            best_params = x0
            best_side = side
        else:
            # Last resort: return a dummy valid config (e.g. all in a line, huge side)
            # This will score poorly but be valid.
            p = np.zeros(3*N)
            for i in range(N):
                p[3*i] = i * 2.0
                p[3*i+1] = 0.0
                p[3*i+2] = 0.0
            best_params = p
            best_side = compute_min_outer_side(p, (0.0, 0.0), 0.0)
            
    # Extract positions and angles from best_params
    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], angles[i]])
    
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
