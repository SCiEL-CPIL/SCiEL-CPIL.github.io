import numpy as np
import time
import random
import math
from scipy.optimize import linear_sum_assignment

N = 11
DEADLINE = 145.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """
    Compute vertices of a regular hexagon.
    side: length of the side.
    angle_deg: rotation of the hexagon.
    """
    a = np.radians(angle_deg)
    # Circumradius R = side
    # Inradius r = side * sqrt(3) / 2
    # Vertices at angles 0, 60, ... relative to center, rotated by a.
    # Standard orientation: flat top? Pointy top?
    # Let's use vertices at 0, 60, 120...
    # This corresponds to "pointy" left/right if angle=0.
    # Wait, if vertices are at 0 deg, the side is vertical? No.
    # Vertex 0: (R, 0). Vertex 1: (R/2, R*sqrt(3)/2).
    # Side between 0 and 1 has length R.
    # So this is a hexagon with vertices at angles 0, 60...
    
    angles = np.arange(6) * np.pi / 3 + a
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_face_normals(angle_deg):
    """
    Normals to the 6 faces of the hexagon.
    For a hexagon with vertices at 0, 60..., the faces are centered at 30, 90...
    So normals are at 30, 90, 150...
    """
    a = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3 + np.pi / 6 + a
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _min_enclosing_side(centers, angles, inner_side=1.0):
    """
    Compute the minimum side length S of a regular hexagon (centered at origin, 
    angle 0 for simplicity, or we can optimize angle) that encloses the given 
    inner hexagons.
    
    Actually, the problem allows the outer hexagon to rotate and translate.
    However, by symmetry, we can fix the outer hexagon's center at (0,0) 
    and optimize its angle? 
    Or better: The score is the outer side length.
    We can fix the outer hexagon orientation to 0 (or any fixed angle) 
    and allow inner hexagons to rotate? 
    No, the outer hexagon can rotate. 
    But since the inner hexagons can rotate arbitrarily, the relative 
    orientation matters.
    
    Let's fix the outer hexagon at center (0,0) and angle 0.
    Then we just need to find the minimum S such that all inner hexagons 
    are inside.
    For a fixed outer hexagon (angle 0), the constraint for an inner hexagon 
    at (cx, cy, ang) is that its support function in the 6 outer normal 
    directions must be <= S * sqrt(3) / 2.
    
    Support function of inner hex i in direction n:
    h_i(n) = max_{v in verts_i} dot(v, n)
    Since the outer hexagon is centered at origin, the condition is:
    max_{v in verts_i} dot(v, n_k) <= S * sqrt(3) / 2 for all k=0..5.
    
    So S >= 2/sqrt(3) * max_{i, k} dot(verts_i, n_k_outer).
    
    We can also optimize the outer angle. Let's try optimizing the outer angle 
    along with the inner positions.
    But to keep the CSP simple, let's fix outer angle to 0 for now, 
    or try a few fixed angles.
    Actually, optimizing outer angle is a 1D problem. We can do a grid search 
    or optimization on outer_angle.
    
    For the backtracking, let's fix outer_angle = 0.
    If we find a solution, we can try to shrink it by optimizing outer_angle.
    """
    outer_normals = _hex_face_normals(0.0)
    max_proj = 0.0
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i], inner_side)
        projs = verts @ outer_normals.T
        max_proj = max(max_proj, np.max(projs))
    
    # S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def _are_disjoint(v1, v2, tol=1e-9):
    """
    Check if two convex polygons (hexagons) are disjoint using SAT.
    v1, v2: (6, 2) arrays of vertices.
    """
    # Get edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        edge = p2 - p1
        # Normal to edge
        n = np.array([-edge[1], edge[0]])
        # Project v1 and v2 onto n
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True # Separated on this axis
    
    # Check edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
            
    return False # Overlap

def _can_place(centers, angles, idx, cx, cy, ang, tol=1e-9):
    """
    Check if placing a hexagon at (cx, cy, ang) at index idx 
    is disjoint from all previously placed hexagons.
    """
    new_verts = _hex_vertices(cx, cy, ang, 1.0)
    for i in range(idx):
        old_verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
        if not _are_disjoint(new_verts, old_verts, tol):
            return False
    return True

def _solve_backtracking(target_side, outer_angle=0.0):
    """
    Try to place N hexagons inside a hexagon of side `target_side`.
    Uses backtracking with a grid of candidate positions.
    """
    # Inradius of outer hexagon
    R_in = target_side * np.sqrt(3) / 2.0
    
    # Generate candidate positions
    # The centers must be within distance R_in - 1 (approx) from center?
    # Actually, the farthest point of the inner hexagon from its center is 1.
    # So the center must be within the "eroded" outer hexagon.
    # Erosion of a hexagon by a circle of radius 1? No, by a hexagon.
    # The set of valid centers is the Minkowski difference of Outer and Inner.
    # This is a smaller hexagon.
    # Side length of eroded hexagon: S_outer - 2 * R_inner? No.
    # The inradius reduces by the inradius of the inner hexagon?
    # If we translate the inner hexagon, the constraint is on the vertices.
    # The locus of centers is a hexagon with side S_outer - 2 (if aligned)?
    # Let's just use a grid within the bounding box of the eroded region.
    
    # Approximate max radius for center:
    # Max distance from center to a point in outer hex is S_outer.
    # Min distance from center to boundary of inner hex is sqrt(3)/2.
    # So center can be at most S_outer - sqrt(3)/2?
    # Actually, if the inner hexagon is aligned, the max projection is S_outer*sqrt(3)/2.
    # Inner hex max projection is 1 (circumradius).
    # So center projection must be <= S_outer*sqrt(3)/2 - 1.
    # So the eroded region is a hexagon with inradius R_in - 1?
    # No, the support function of the Minkowski difference is h_outer - h_inner.
    # For a hexagon, h_outer(n) = R_in * cos(pi/6) / cos(angle_diff)?
    # Simpler: The eroded region is a hexagon with side length S_outer - 2?
    # Let's verify.
    # If S_outer = 2, inradius = sqrt(3).
    # Inner hex inradius = sqrt(3)/2.
    # If we place inner hex at center, it fits.
    # If we move it to the edge, the vertex touches.
    # The distance from center to the "eroded" boundary along the normal is R_in - 1?
    # No, the inradius of the inner hex is sqrt(3)/2.
    # The max extent of the inner hex in the normal direction is 1 (circumradius).
    # So the center can be at distance R_in - 1 from the outer boundary along the normal?
    # Yes, if the vertex points outward.
    # So the valid region for centers is a hexagon with inradius R_in - 1.
    # Side length of this region: (R_in - 1) * 2 / sqrt(3).
    
    R_valid_in = R_in - 1.0
    if R_valid_in < 0:
        return None
        
    S_valid = R_valid_in * 2 / np.sqrt(3)
    
    # Grid resolution
    # We want to try positions.
    # Let's create a grid of candidate centers and angles.
    # Angles: 0, 30, 60, 90, 120, 150. (6-fold symmetry, so 0-180 is enough, but 0-60 covers it if we consider reflection? No, rotation is continuous. But hexagons have 60 deg symmetry. So 0-60 is enough for relative orientation? No, absolute orientation matters for the grid? No, the outer is fixed. The inner can rotate. The set of valid orientations is continuous. But due to symmetry of the hexagon shape, rotating by 60 degrees yields the same shape. So we only need to check angles in [0, 60).
    # Let's sample angles: 0, 15, 30, 45.
    angles_candidates = [0.0, 15.0, 30.0, 45.0]
    
    # Grid of centers
    # Range: -S_valid to S_valid in x and y?
    # The hexagon is not a square.
    # Let's use a hexagonal grid or a fine square grid.
    # Square grid is easier.
    # Max extent in x: S_valid * cos(30) = S_valid * sqrt(3)/2?
    # Vertices of valid region:
    # (S_valid, 0), (S_valid/2, S_valid*sqrt(3)/2), ...
    # So x in [-S_valid, S_valid], y in [-S_valid*sqrt(3)/2, S_valid*sqrt(3)/2].
    
    x_max = S_valid
    y_max = S_valid * np.sqrt(3) / 2
    
    # Step size
    # We want to cover the area.
    # Area of valid region: 3 * sqrt(3) / 2 * S_valid^2.
    # Number of hexagons to place: 11.
    # They are disjoint. Area of one hex: 3 * sqrt(3) / 2.
    # Total area needed: 11 * 3 * sqrt(3) / 2.
    # Density: 11 / (S_valid^2).
    # If S_valid is small, grid must be fine.
    
    # Let's use a step size of 0.2 initially.
    step = 0.2
    if S_valid < 3.0:
        step = 0.1
    elif S_valid < 5.0:
        step = 0.15
    
    xs = np.arange(-x_max, x_max + step, step)
    ys = np.arange(-y_max, y_max + step, step)
    
    # Generate all candidates (x, y, angle)
    candidates = []
    for x in xs:
        for y in ys:
            # Check if (x,y) is inside the valid hexagon
            # Valid hexagon: |x| <= S_valid, |y| <= S_valid*sqrt(3)/2, |x| + |y|/sqrt(3) <= S_valid?
            # Equation of hexagon side: x * cos(30) + y * sin(30) <= S_valid * cos(30)? No.
            # Inradius R_valid_in.
            # Distance to boundary: R_valid_in - (|x| * cos(pi/6) + |y| * sin(pi/6))? No.
            # Let's just check if the hexagon centered at (x,y) fits in the outer.
            # We can do a quick check using support functions.
            # But for generating candidates, a bounding box is enough.
            # We will filter during placement.
            
            for ang in angles_candidates:
                candidates.append((x, y, ang))
                
    # Shuffle candidates to randomize the order (helps with backtracking)
    random.shuffle(candidates)
    
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Backtracking
    # We place hexagons one by one.
    # To improve efficiency, we can use a "Best-Fit" strategy:
    # Sort candidates by some heuristic? 
    # Or just try them in random order.
    # Given the small N, random order with restarts is robust.
    
    def backtrack(idx):
        if idx == N:
            return True
            
        for cx, cy, ang in candidates:
            if _can_place(centers, angles, idx, cx, cy, ang):
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                angles[idx] = ang
                
                if backtrack(idx + 1):
                    return True
                    
                # Unplace
                # No need to explicitly unset if we overwrite
        return False
        
    if backtrack(0):
        return centers, angles
    return None

def _optimize_outer_angle(centers, angles):
    """
    Given a valid configuration of inner hexagons, find the outer hexagon
    angle that minimizes the side length.
    """
    # The side length S(theta) = 2/sqrt(3) * max_{i, k} dot(verts_i, n_k(theta))
    # n_k(theta) are the normals of the outer hexagon at angle theta.
    # This is a function of theta. We can minimize it.
    
    def calc_side(theta_deg):
        outer_normals = _hex_face_normals(theta_deg)
        max_proj = 0.0
        for i in range(N):
            verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
            projs = verts @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
        return 2.0 * max_proj / np.sqrt(3)
        
    # Grid search for theta
    best_theta = 0.0
    best_side = float('inf')
    for theta in np.arange(0, 60, 1.0):
        s = calc_side(theta)
        if s < best_side:
            best_side = s
            best_theta = theta
            
    # Refine
    for theta in np.arange(best_theta - 1, best_theta + 1, 0.1):
        s = calc_side(theta)
        if s < best_side:
            best_side = s
            best_theta = theta
            
    return best_side, best_theta

def _check_validity(inner, center, side, angle, tol=1e-5):
    """
    Strict check for the final result.
    inner: list of (x, y, angle_deg)
    center: (x, y)
    side: float
    angle: float
    """
    cx, cy = center
    outer_verts = _hex_vertices(cx, cy, angle, side)
    outer_normals = _hex_face_normals(angle)
    inradius = side * np.sqrt(3) / 2.0
    
    # 1. Containment
    for i in range(N):
        ix, iy, iang = inner[i]
        verts = _hex_vertices(ix, iy, iang, 1.0)
        # Check if all vertices are inside outer hex
        # Using support functions is faster for convex polygons
        # But let's check vertices against faces for robustness
        for n in outer_normals:
            # Project vertices onto n
            rel = verts - np.array([cx, cy])
            projs = rel @ n
            if np.max(projs) > inradius + tol:
                return False
                
    # 2. Disjointness
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
            v2 = _hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if not _are_disjoint(v1, v2, tol):
                return False
                
    return True

def _generate_initial_config(side):
    """
    Generate a rough initial configuration to help guide the search?
    The backtracking doesn't use this, but we can use it to set a target.
    """
    return None

def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_side = 10.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Binary search on side length
    # Lower bound: Area argument.
    # Area outer = 3*sqrt(3)/2 * S^2
    # Area inner = 11 * 3*sqrt(3)/2
    # S^2 >= 11 => S >= sqrt(11) ~ 3.31
    # But packing efficiency is < 1.
    # For hexagons, max density is 1 (if aligned).
    # So S >= sqrt(11) is a theoretical lower bound.
    # In practice, due to boundary effects, S is larger.
    # Let's start with lo=4.0, hi=6.0
    
    lo = 4.0
    hi = 6.5
    
    # We will try to find a valid configuration for a given side.
    # If found, we shrink. If not, we expand.
    
    current_side = 5.5
    attempts = 0
    max_attempts = 100
    
    while _remaining_time() > 5.0 and attempts < max_attempts:
        attempts += 1
        
        found = False
        # Try multiple random restarts for the backtracking
        for restart in range(20):
            if _remaining_time() < 5.0:
                break
                
            # Randomize candidate order inside _solve_backtracking is already done
            # But we can also randomize the grid slightly?
            # No, the grid is deterministic. The shuffle is the randomization.
            
            res = _solve_backtracking(current_side)
            if res is not None:
                centers, angles = res
                # Optimize outer angle
                opt_side, opt_angle = _optimize_outer_angle(centers, angles)
                
                # Check validity with the optimized side and angle
                # Note: _optimize_outer_angle returns the side length required.
                # If opt_side <= current_side, it's valid for current_side.
                # But we want the MINIMUM side.
                # So we update best_side with opt_side.
                
                inner_list = []
                for i in range(N):
                    inner_list.append((centers[i, 0], centers[i, 1], angles[i]))
                
                # Verify validity with the calculated side
                if _check_validity(inner_list, (0.0, 0.0), opt_side, opt_angle):
                    if opt_side < best_side:
                        best_side = opt_side
                        best_inner = inner_list
                        best_center = (0.0, 0.0)
                        best_angle = opt_angle
                        
                    found = True
                    break # Found a valid config for this side, try to shrink
        
        if found:
            # Shrink
            current_side *= 0.98
            if current_side < lo:
                current_side = lo
        else:
            # Expand
            current_side *= 1.02
            if current_side > hi:
                current_side = hi
                # Reset if stuck?
                random.seed(attempts)
                np.random.seed(attempts)
                
    if best_inner is None:
        # Fallback: Return a valid configuration with a large side
        # Place them in a circle
        inner_list = []
        r = 4.0
        for i in range(N):
            a = 2 * np.pi * i / N
            x = r * np.cos(a)
            y = r * np.sin(a)
            inner_list.append((x, y, 0.0))
        
        # Calculate required side
        # Just use a large side
        side = 8.0
        return inner_list, (0.0, 0.0), side, 0.0
        
    return best_inner, best_center, best_side, best_angle

if __name__ == "__main__":
    res = construct()
    print(res)
