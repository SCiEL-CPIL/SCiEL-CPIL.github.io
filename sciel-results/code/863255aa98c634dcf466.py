"""
Rigid-body topology-driven hexagon packing using smooth gradient-based optimization.
Strategy:
1. Define a fixed adjacency topology (center + ring 1 + ring 2) for 11 hexagons.
2. Use smooth differentiable penalties for overlap (based on distance between centers)
   and for containment (based on support functions).
3. Optimize positions and angles using scipy.optimize.minimize (L-BFGS-B) from multiple
   random initializations to escape local minima.
4. Binary search or direct optimization of the outer container side length is handled
   by minimizing the max distance from the center to any vertex, which corresponds to
   the required side length for a hexagon centered at the origin.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.asarray(inner_angles, dtype=np.float64)
    
    # Generate all vertices for all inner hexagons
    all_verts = []
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Rotate vertices into the outer hexagon's coordinate frame
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # The condition for a point (lx, ly) to be inside a hexagon of side S centered at origin
    # is max(|lx|, |lx + ly/sqrt(3)|, |lx - ly/sqrt(3)|) <= S
    # Actually, the support function in the 6 directions must be <= S.
    # Directions are at 0, 60, 120, 180, 240, 300 degrees.
    # In the rotated frame, the hexagon sides are aligned with axes? 
    # Standard hexagon with flat top/bottom (angle 0) has vertices at (S, 0), (S/2, S*sqrt(3)/2), etc.
    # The inequalities are:
    # |x| <= S
    # |x/2 + y*sqrt(3)/2| <= S
    # |-x/2 + y*sqrt(3)/2| <= S
    # Which simplifies to max(|x|, |x + y/sqrt(3)|, |x - y/sqrt(3)|) <= S? 
    # Let's check:
    # If we use the form: max(|lx|, |lx + ly/SQRT3|, |lx - ly/SQRT3|) <= S
    # Let's verify with a vertex at (S, 0): max(S, |S|, |S|) = S. Correct.
    # Vertex at (S/2, S*sqrt(3)/2): 
    #   lx = S/2, ly = S*sqrt(3)/2
    #   lx + ly/sqrt(3) = S/2 + S/2 = S
    #   lx - ly/sqrt(3) = S/2 - S/2 = 0
    #   max(S/2, S, 0) = S. Correct.
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / SQRT3)
    c3 = np.abs(lx - ly / SQRT3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check overlaps using SAT
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            v1 = verts_list[i]
            v2 = verts_list[j]
            # Generate axes from edges of v1 and v2
            axes = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    nx, ny = edge[1], -edge[0]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm > 1e-12:
                        axes.append(np.array([nx/norm, ny/norm]))
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 - tol or max2 < min1 - tol:
                    overlap = False
                    break
            
            if overlap:
                return False
    
    # Check containment
    for verts in verts_list:
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            a = np.radians(outer_angle)
            ca, sa = np.cos(a), np.sin(a)
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            
            c1 = abs(lx)
            c2 = abs(lx + ly / SQRT3)
            c3 = abs(lx - ly / SQRT3)
            if max(c1, c2, c3) > outer_side + tol:
                return False
    return True

def _objective(params):
    """
    Objective function for scipy.optimize.minimize.
    params: array of length 2*N + 1 (x1, y1, a1, x2, y2, a2, ..., S)
    """
    # Unpack
    x = params[0::3]
    y = params[1::3]
    a = params[2::3]
    S = params[-1]  # Outer side length
    
    # Penalty for S being too small (containment)
    # Calculate required S for current positions
    pos = np.column_stack([x, y])
    req_S = _min_outer_side(pos, a, (0.0, 0.0), 0.0)
    containment_penalty = max(0, req_S - S) ** 2
    
    # Penalty for overlaps
    # Use smooth approximation of overlap penalty based on distance between centers
    # For two unit hexagons with relative angle delta, the minimum distance for non-overlap is 2.0 (if aligned)
    # Actually, the distance between centers must be at least the sum of support functions in the direction of separation.
    # A simpler smooth penalty: for each pair, if distance < 2.0, add penalty.
    # But rotation matters. Let's use a more robust smooth penalty based on vertices.
    # However, vertex-based penalties are non-smooth at intersections.
    # Let's use the distance between centers as a proxy for overlap, assuming worst-case rotation?
    # No, that's not tight.
    
    # Better: Use the exact minimum distance between polygons? Too slow for gradient.
    # Alternative: Use a soft penalty based on the overlap area approximation or just center distance with a large weight.
    # Given the topology is fixed (hexagonal grid), the optimal solution has centers at specific distances.
    # In a perfect hexagonal packing, adjacent centers are distance 2 apart.
    # We can enforce that adjacent centers are at least 2.0 apart.
    
    overlap_penalty = 0.0
    # Define adjacency pairs based on the hexagonal grid topology
    # Indices: 0 (center), 1-6 (ring 1), 7-10 (ring 2 subset)
    # Wait, N=11. Center + 6 + 4? 
    # The standard hexagonal cluster of 11 is Center + Ring 1 (6) + Ring 2 (4 corners? No, Ring 2 has 12, we take 4?)
    # Actually, the "flower" of 7 is Center + 6.
    # 11 hexagons: Center + 6 (Ring 1) + 4 (from Ring 2, specifically the ones opposite to gaps?)
    # Let's stick to the initial packing structure from the seed:
    # 0: (0,0)
    # 1-6: distance 2, angles 0, 60, ...
    # 7-10: distance 4, angles 0, 60, 120, 180 (4 hexagons)
    
    # Adjacencies in this specific cluster:
    # 0 is adjacent to 1,2,3,4,5,6
    # 1 is adjacent to 0, 2, 6, 7 (if 7 is at angle 0, dist 4? No, 7 is at (4,0). 1 is at (2,0). Dist 2. Yes.)
    # Let's just check all pairs with a soft penalty if distance < 2.0.
    # This is a heuristic but effective for hexagonal packings.
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            # Minimum distance for non-overlapping unit hexagons is 2.0 (when aligned)
            # With rotation, it can be slightly less, but 2.0 is a good lower bound for "aligned" hexagons.
            # If we allow rotation, the minimum distance can be smaller.
            # Let's use a threshold of 1.9 to be safe, or just use 2.0 and rely on the optimizer to find valid rotations.
            # Actually, if two hexagons are rotated 30 degrees relative to each other, they can be closer.
            # To be robust, we should check the actual polygon distance.
            # But for speed, let's use a soft penalty on center distance < 2.0.
            if dist < 2.0:
                overlap_penalty += (2.0 - dist) ** 2
    
    # Regularization to keep S small
    return S + 100.0 * containment_penalty + 1000.0 * overlap_penalty

def _construct():
    global START_TIME, DEADLINE
    
    # Initial guess based on hexagonal grid
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    a0 = np.zeros(N)
    S0 = 5.0
    
    # Set initial positions
    # 0: center
    x0[0] = 0.0
    y0[0] = 0.0
    
    # 1-6: ring 1
    for i in range(1, 7):
        ang = (i - 1) * np.pi / 3
        x0[i] = 2.0 * np.cos(ang)
        y0[i] = 2.0 * np.sin(ang)
    
    # 7-10: ring 2 (4 hexagons)
    for i in range(4):
        ang = i * np.pi / 3
        x0[7 + i] = 4.0 * np.cos(ang)
        y0[7 + i] = 4.0 * np.sin(ang)
    
    # Initial params: x, y, a, S
    # Interleave: x1, y1, a1, x2, y2, a2, ...
    params0 = np.zeros(3 * N + 1)
    for i in range(N):
        params0[3*i] = x0[i]
        params0[3*i+1] = y0[i]
        params0[3*i+2] = a0[i]
    params0[-1] = S0
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Multi-start optimization
    n_restarts = 10
    time_per_restart = 10.0
    
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10.0:
            break
        
        # Perturb initial guess
        if r > 0:
            noise_scale = 0.5
            perturbed = params0.copy()
            for i in range(N):
                perturbed[3*i] += random.gauss(0, noise_scale)
                perturbed[3*i+1] += random.gauss(0, noise_scale)
                perturbed[3*i+2] += random.gauss(0, 0.2)
            perturbed[-1] = S0 + random.gauss(0, 0.5)
        else:
            perturbed = params0.copy()
        
        try:
            res = minimize(_objective, perturbed, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8, 'maxfun': 1000})
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best solution
    x = best_params[0::3]
    y = best_params[1::3]
    a = best_params[2::3]
    S = best_params[-1]
    
    # Refine S to be the exact minimum required
    pos = np.column_stack([x, y])
    exact_S = _min_outer_side(pos, a, (0.0, 0.0), 0.0)
    
    # Check validity
    inner_list = [[float(x[i]), float(y[i]), float(a[i])] for i in range(N)]
    
    if _check_validity(inner_list, (0.0, 0.0), exact_S, 0.0):
        return inner_list, (0.0, 0.0), float(exact_S), 0.0
    else:
        # Fallback to initial grid
        pos_init = np.column_stack([x0, y0])
        ang_init = a0
        S_init = _min_outer_side(pos_init, ang_init, (0.0, 0.0), 0.0)
        inner_init = [[float(x0[i]), float(y0[i]), float(ang_init[i])] for i in range(N)]
        return inner_init, (0.0, 0.0), float(S_init), 0.0

def construct():
    return _construct()
