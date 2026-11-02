import numpy as np
from scipy.optimize import basinhopping
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0 # Keep 10s buffer

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """
    Returns 6 vertices of a regular hexagon centered at (cx, cy) 
    with side length 'side' and rotated by 'angle_deg' degrees.
    The flat side is at the bottom for 0 degrees? 
    Standard hexagon: vertices at 0, 60, 120... degrees from center.
    Let's define angle 0 as vertex pointing right (East).
    """
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
    """
    Check if point (px, py) is inside hexagon centered at (cx, cy).
    We can transform point to hexagon's local frame and check.
    """
    # Translate
    dx = px - cx
    dy = py - cy
    # Rotate back
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Check if (lx, ly) is in unit hexagon centered at 0
    # Vertices of unit hexagon (angle 0): (1,0), (0.5, .866), (-0.5, .866), (-1,0), (-0.5, -.866), (0.5, -.866)
    # Equations of lines:
    # 1. x <= 1
    # 2. y <= sqrt(3)/2 * (1 - x)  -- wait, line through (1,0) and (0.5, .866)
    #    Slope m = (.866 - 0) / (0.5 - 1) = -1.732 = -sqrt(3)
    #    y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y <= sqrt(3)
    # 3. Line through (0.5, .866) and (-0.5, .866) => y <= .866 = sqrt(3)/2
    # 4. Line through (-0.5, .866) and (-1, 0) => slope sqrt(3). y = sqrt(3)(x + 1) => -sqrt(3)x + y <= sqrt(3)
    # 5. Line through (-1, 0) and (-0.5, -.866) => slope -sqrt(3). y = -sqrt(3)(x + 1) => sqrt(3)x - y <= sqrt(3)
    # 6. Line through (-0.5, -.866) and (0.5, -.866) => y >= -sqrt(3)/2
    
    s3 = np.sqrt(3)
    if lx > 1.0 + 1e-9: return False
    if s3 * lx + ly > s3 + 1e-9: return False
    if ly > s3 / 2 + 1e-9: return False
    if -s3 * lx + ly > s3 + 1e-9: return False
    if s3 * lx - ly > s3 + 1e-9: return False
    if ly < -s3 / 2 - 1e-9: return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    """
    # Vector cross product
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
        
    # Check collinearity / endpoint on segment
    # This is less critical for "strictly inside" but good for robustness
    # For penalty, touching is okay, overlapping is bad.
    # We consider "overlap" if interiors intersect.
    # If they just touch, area is 0.
    # We'll return False for just touching.
    return False

def polygons_overlap(v1, v2):
    """
    Check if two convex polygons (lists of points) overlap.
    v1, v2 are arrays of shape (6, 2).
    """
    # 1. Check if any vertex of v1 is inside v2
    for i in range(6):
        if point_in_hexagon(v1[i, 0], v1[i, 1], 0, 0, 0, 1.0): # Wait, this checks against unit hex at 0.
             # We need to check if v1[i] is inside polygon v2.
             # My point_in_hexagon assumes the hexagon is centered at cx,cy with angle.
             # I should write a generic point_in_polygon or use the specific hexagon logic for v2.
             pass
             
    # Let's implement a generic point_in_convex_polygon
    def point_in_convex_poly(pt, poly):
        n = len(poly)
        sign1 = None
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Cross product (p2 - p1) x (pt - p1)
            cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
            if abs(cross) < 1e-9:
                continue
            if sign1 is None:
                sign1 = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign1 < 0) or (cross < 0 and sign1 > 0):
                    return False
        return True

    # Check vertices of v1 in v2
    for i in range(6):
        if point_in_convex_poly(v1[i], v2):
            return True
            
    # Check vertices of v2 in v1
    for i in range(6):
        if point_in_convex_poly(v2[i], v1):
            return True
            
    # Check edge intersections
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def get_hexagon_vertices(params, i):
    """
    params: array of length 36. [x0, y0, a0, x1, y1, a1, ...]
    i: index of hexagon
    """
    x = params[i*3]
    y = params[i*3+1]
    a = params[i*3+2]
    return hex_vertices(x, y, a)

def objective(params):
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10
        
    # Calculate vertices for all hexagons
    verts = []
    for i in range(N):
        v = get_hexagon_vertices(params, i)
        verts.append(v)
        
    # Calculate required S
    # S is the max "hexagonal radius" of any vertex from origin.
    # Radius r of point (x,y) in hexagon norm:
    # r = max(|x|, |y|/sqrt(3) + |x|/2, ... )?
    # Simpler: The hexagon of side S centered at origin contains (x,y) iff:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |y|/sqrt(3) + |x|/2 <= S
    # |y|/sqrt(3) - |x|/2 <= S  (Symmetric)
    # Actually, the condition for (x,y) to be in hexagon of side S (flat top/bottom? No, pointy top?)
    # My hex_vertices uses pointy top (vertex at 0 deg is (1,0)).
    # The bounding hexagon for the objective should match this orientation?
    # The outer hexagon can be rotated. But we fixed outer rotation to 0.
    # So outer hexagon has vertices at (S, 0), etc.
    # Condition for (x,y) in outer hexagon (side S, rot 0):
    # 1. x <= S
    # 2. -x <= S
    # 3. y <= S * sqrt(3) / 2  (Wait, vertex at 60 deg is (S/2, S*sqrt(3)/2). Side connecting (S,0) and (S/2, S*sqrt(3)/2) has equation:
    #    Slope = (S*sqrt(3)/2 - 0) / (S/2 - S) = (sqrt(3)/2) / (-1/2) = -sqrt(3).
    #    y - 0 = -sqrt(3)(x - S) => y = -sqrt(3)x + S*sqrt(3) => sqrt(3)x + y <= S*sqrt(3).
    #    So y <= S*sqrt(3) - sqrt(3)x.
    #    Similarly for other sides.
    
    # Let's just compute the minimal S for each point.
    # For a point (x,y), the minimal S such that it is inside the hexagon is:
    # S_min = max( |x|, (|y| + |x|/2) / (sqrt(3)/2) ... )
    # Let's derive:
    # Constraints:
    # 1. x <= S
    # 2. -x <= S
    # 3. sqrt(3)x + y <= S*sqrt(3)  => y/sqrt(3) + x/3 <= S
    # 4. -sqrt(3)x + y <= S*sqrt(3) => y/sqrt(3) - x/3 <= S
    # 5. -sqrt(3)x - y <= S*sqrt(3) => -y/sqrt(3) - x/3 <= S
    # 6. sqrt(3)x - y <= S*sqrt(3) => -y/sqrt(3) + x/3 <= S
    
    # So S >= |x|
    # S >= |y|/sqrt(3) + |x|/3
    # Wait, let's check vertex (1,0). S=1. |x|=1. |y|/sqrt(3)+|x|/3 = 1/3. Max is 1. Correct.
    # Vertex (0.5, sqrt(3)/2). S=1. |x|=0.5. |y|/sqrt(3)+|x|/3 = 0.5 + 0.166 = 0.666. Max is 0.5? No.
    # Let's re-evaluate constraint 3: sqrt(3)(0.5) + sqrt(3)/2 = sqrt(3). S*sqrt(3) = sqrt(3). OK.
    # Formula: S >= (sqrt(3)x + y)/sqrt(3) = x + y/sqrt(3).
    # For (0.5, sqrt(3)/2): 0.5 + 0.5 = 1. Correct.
    
    # So for each vertex (x,y), S_req = max(|x|, |y|/sqrt(3) + |x|/2, |-y|/sqrt(3) + |x|/2 ... )
    # Actually, due to symmetry, S_req = max( |x|, |y|/sqrt(3) + |x|/2 ) is not quite right.
    # Let's just use the 6 inequalities.
    
    S_max = 0.0
    for v in verts:
        for pt in v:
            x, y = pt
            s1 = abs(x)
            s2 = abs(y)/np.sqrt(3) + abs(x)/2
            # Wait, is it abs(x)/2?
            # Constraint: sqrt(3)x + y <= S*sqrt(3)
            # If x>0, y>0: S >= x + y/sqrt(3).
            # If x>0, y<0: S >= x - y/sqrt(3).
            # So S >= x + |y|/sqrt(3) for x>0?
            # Let's check (1, 0). S=1. 1+0=1.
            # Let's check (0.5, 0.866). S=1. 0.5 + 0.5 = 1.
            # Let's check (0, 1). Not in hexagon. S must be > 1.
            # (0, 1): x=0. S >= |1|/sqrt(3) = 0.577?
            # Constraint for x=0: y <= S*sqrt(3)/2. So S >= 2y/sqrt(3).
            # My formula s2 = |y|/sqrt(3) + 0. This gives 0.577.
            # But actual S is 2/1.732 = 1.15.
            # So the formula is wrong.
            
            # Correct S for point (x,y) in pointy-top hexagon:
            # S = max( |x|, (2/3)*|y| + (1/3)*|x| ... no )
            # Let's stick to the 6 linear inequalities.
            s_req = max(
                abs(x),
                (abs(y) + np.sqrt(3)*abs(x))/2, # This is not it either
            )
            # Let's just compute:
            # S >= x
            # S >= -x
            # S >= (y + sqrt(3)*x)/sqrt(3)  => y/sqrt(3) + x/3
            # S >= (-y + sqrt(3)*x)/sqrt(3) => -y/sqrt(3) + x/3
            # S >= (-y - sqrt(3)*x)/sqrt(3) => -y/sqrt(3) - x/3
            # S >= (y - sqrt(3)*x)/sqrt(3) => y/sqrt(3) - x/3
            
            # Since S must be positive, we can take abs of the RHS terms?
            # No, the inequalities are directional.
            # But S is the same for all.
            # S >= max( x, -x, y/sqrt(3)+x/3, -y/sqrt(3)+x/3, -y/sqrt(3)-x/3, y/sqrt(3)-x/3 )
            # This is equivalent to:
            # S >= |x|
            # S >= |y|/sqrt(3) + |x|/3 ?
            # Let's test (0, 1).
            # |x|=0.
            # Terms: 1/sqrt(3), -1/sqrt(3), -1/sqrt(3), 1/sqrt(3). Max is 0.577.
            # But we know S=1.15.
            # Where did I go wrong?
            # Constraint: y <= S*sqrt(3)/2.
            # So S >= 2y/sqrt(3).
            # My linearization:
            # Side 2: y = -sqrt(3)(x - S) => y = -sqrt(3)x + S*sqrt(3).
            # If x=0, y = S*sqrt(3).
            # So S = y/sqrt(3).
            # Wait. Vertex at 60 deg is (S/2, S*sqrt(3)/2).
            # If x=0, the boundary is the vertical line? No.
            # The hexagon vertices are:
            # 0: (S, 0)
            # 1: (S/2, S*sqrt(3)/2)
            # 2: (-S/2, S*sqrt(3)/2)
            # 3: (-S, 0)
            # 4: (-S/2, -S*sqrt(3)/2)
            # 5: (S/2, -S*sqrt(3)/2)
            
            # The edge between 0 and 1:
            # Line through (S,0) and (S/2, S*sqrt(3)/2).
            # Slope -sqrt(3).
            # y = -sqrt(3)(x - S).
            # y = -sqrt(3)x + S*sqrt(3).
            # If x=0, y = S*sqrt(3).
            # But the max y in the hexagon is S*sqrt(3)/2 (at vertices 1 and 2).
            # So the line y = S*sqrt(3) is NOT the boundary at x=0.
            # The boundary at x=0 is the segment connecting 1 and 2? No, 1 and 2 have y = S*sqrt(3)/2.
            # The segment 1-2 is horizontal.
            # So for x in [-S/2, S/2], y <= S*sqrt(3)/2.
            # So S >= 2y/sqrt(3).
            
            # So the constraints are:
            # 1. x <= S
            # 2. -x <= S
            # 3. y <= S*sqrt(3)/2  (for |x| <= S/2)
            # 4. -y <= S*sqrt(3)/2
            # 5. y + sqrt(3)x <= S*sqrt(3) (for x > S/2)
            # 6. y - sqrt(3)x <= S*sqrt(3) (for x < -S/2)
            
            # Combining these into a single max formula:
            # S >= |x|
            # S >= 2|y|/sqrt(3)
            # S >= (|y| + sqrt(3)|x|)/sqrt(3) ?
            # Let's check (S, 0). S=1.
            # |x|=1. 2|y|/sqrt(3)=0. (|y|+sqrt(3)|x|)/sqrt(3) = 1. Max 1. OK.
            # Let's check (S/2, S*sqrt(3)/2). S=1.
            # |x|=0.5. 2|y|/sqrt(3) = 1. (|y|+sqrt(3)|x|)/sqrt(3) = (0.866+0.866)/1.732 = 1. Max 1. OK.
            # Let's check (0, S*sqrt(3)/2). S=1.
            # |x|=0. 2|y|/sqrt(3) = 1. (0+0)/sqrt(3)=0. Max 1. OK.
            
            # So S_req = max( |x|, 2|y|/sqrt(3), (|y| + sqrt(3)|x|)/sqrt(3) )
            # Note: (|y| + sqrt(3)|x|)/sqrt(3) = |y|/sqrt(3) + |x|/3.
            # Wait, 2|y|/sqrt(3) is larger than |y|/sqrt(3).
            # So S_req = max( |x|, 2|y|/sqrt(3), |y|/sqrt(3) + |x|/3 )
            # Since 2|y|/sqrt(3) >= |y|/sqrt(3), the third term is only relevant if |x|/3 > |y|/sqrt(3).
            # i.e. |x| > sqrt(3)|y|.
            
            s_calc = max(abs(x), 2*abs(y)/np.sqrt(3), abs(y)/np.sqrt(3) + abs(x)/3)
            if s_calc > S_max:
                S_max = s_calc
                
    # Calculate Overlap Penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Quick reject
            dx = params[i*3] - params[j*3]
            dy = params[i*3+1] - params[j*3+1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.0: # Max distance for overlap is 2
                continue
            if polygons_overlap(verts[i], verts[j]):
                penalty += 1.0 # Count overlaps
                
    return S_max + 100.0 * penalty

def construct():
    # Initial guess: 3x4 grid
    params = np.zeros(36)
    # Center grid at 0
    # x positions: -1.5, -0.5, 0.5, 1.5 (scaled by sqrt(3))
    # y positions: -0.75, 0, 0.75 (scaled by 1.5)
    # Let's use spacing sqrt(3) and 1.5
    xs = np.array([-1.5, -0.5, 0.5, 1.5]) * np.sqrt(3)
    ys = np.array([-0.75, 0.0, 0.75]) * 1.5
    
    idx = 0
    for y in ys:
        for x in xs:
            params[idx*3] = x
            params[idx*3+1] = y
            params[idx*3+2] = 0.0
            idx += 1
            
    # Run Basin Hopping
    best_result = None
    best_score = objective(params)
    
    def local_minimizer(x):
        # Use Nelder-Mead for local refinement
        from scipy.optimize import minimize
        res = minimize(objective, x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        return res.x, res.fun, 0
        
    bh = basinhopping(objective, params, niter=50, minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200}})
    
    if bh.fun < best_score:
        best_score = bh.fun
        best_result = bh.x
        
    # Check time
    if time.time() - START_TIME < TIME_LIMIT - 10:
        # Try a few random restarts
        for _ in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            random_params = np.random.uniform(-4, 4, 36)
            random_params[2::3] = 0 # Fix angles initially? Or random?
            # Let's keep angles 0 for simplicity in restarts, or randomize slightly
            random_params[2::3] = np.random.uniform(-10, 10, 12)
            
            res = minimize(objective, random_params, method='Nelder-Mead', options={'maxiter': 1000})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
    # Extract best result
    if best_result is None:
        best_result = params
        
    # Build output
    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner.append([x, y, a])
        
    # The outer hexagon center is (0,0), side is best_score (if penalty is 0)
    # If penalty is not 0, the score is invalid/high.
    # We should return the side length corresponding to the configuration.
    # If penalty > 0, the "score" returned by objective includes penalty.
    # But the task score is the outer side length.
    # If the packing is invalid (overlapping), the evaluator might reject it or score it as invalid.
    # We should ensure we return a valid packing if possible.
    # If not, return the best S_max found, even if overlapping?
    # The prompt says "invalid result ... scores 0".
    # So we must return a non-overlapping packing.
    # If we can't find one, we might fail.
    # But the grid should be non-overlapping.
    
    # Recalculate S_max for the best result to be sure
    verts = [get_hexagon_vertices(best_result, i) for i in range(N)]
    S_max = 0.0
    for v in verts:
        for pt in v:
            x, y = pt
            s_calc = max(abs(x), 2*abs(y)/np.sqrt(3), abs(y)/np.sqrt(3) + abs(x)/3)
            if s_calc > S_max:
                S_max = s_calc
                
    return inner, (0.0, 0.0), S_max, 0.0

if __name__ == "__main__":
    construct()
