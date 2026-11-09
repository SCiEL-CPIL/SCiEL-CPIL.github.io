"""Packing unit hexagons in a regular hexagon via multi-start Nelder-Mead."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle):
    """Vertices of regular hexagon centered at (cx,cy), circumradius r, rotation angle (rad)."""
    # Pre-compute cos/sin for speed? No, vectorize inside objective if possible, 
    # but here we return list for clarity, will be used in objective.
    verts = np.zeros((6, 2))
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts[i, 0] = cx + r * math.cos(a)
        verts[i, 1] = cy + r * math.sin(a)
    return verts

def _sat_penalty(v1, v2):
    """
    Compute minimum translation distance to separate two convex polygons (hexagons).
    Returns 0 if they do not overlap.
    Uses Separating Axis Theorem.
    """
    # v1, v2 are (6, 2) arrays
    min_dist = float('inf')
    
    # Axes from v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        
        # Project v1 and v2 onto normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check overlap
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
        
        # Overlap amount
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_dist:
            min_dist = overlap
            
    # Axes from v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_dist:
            min_dist = overlap
            
    return min_dist

def _boundary_penalty(v_inner, center, outer_angle, outer_side):
    """
    Compute penalty for inner hexagon being outside the outer hexagon.
    Returns 0 if inside, else the penetration depth.
    """
    # Outer hexagon face normals
    # Apothem of outer hexagon
    apothem = outer_side * math.sqrt(3) / 2.0
    
    max_pen = 0.0
    for k in range(6):
        a = outer_angle + k * math.pi / 3.0
        normal = np.array([math.cos(a), math.sin(a)])
        
        # Project inner vertices onto normal
        # Relative to center
        rel_verts = v_inner - center
        proj = np.dot(rel_verts, normal)
        
        # Max projection
        max_proj = np.max(proj)
        
        # If max_proj > apothem, it's outside
        if max_proj > apothem:
            pen = max_proj - apothem
            if pen > max_pen:
                max_pen = pen
                
    return max_pen

def objective(params):
    """
    Objective function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ..., xN-1, yN-1, aN-1, cx, cy, outer_angle]
    """
    # Unpack
    positions = params[:3*N].reshape(N, 3)
    cx, cy = params[3*N:3*N+2]
    outer_angle = params[3*N+2]
    
    center = np.array([cx, cy])
    
    # Compute vertices for all inner hexagons
    verts_list = []
    for i in range(N):
        x, y, a = positions[i]
        verts = _hex_vertices(x, y, 1.0, a)
        verts_list.append(verts)
        
    # Compute enclosing side length
    # The side length is determined by the max projection of any vertex onto any outer face normal
    # Normal directions: outer_angle + k*pi/3
    # Projection limit: R * sqrt(3)/2 where R is side length (circumradius)
    # So R = max_proj / (sqrt(3)/2)
    
    max_required_R = 0.0
    for i in range(N):
        verts = verts_list[i]
        rel_verts = verts - center
        for k in range(6):
            a = outer_angle + k * math.pi / 3.0
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(rel_verts, normal)
            max_proj = np.max(proj)
            # R = max_proj / (sqrt(3)/2)
            required_R = max_proj / (math.sqrt(3) / 2.0)
            if required_R > max_required_R:
                max_required_R = required_R
                
    # Ensure minimum size
    if max_required_R < 1.0:
        max_required_R = 1.0
        
    # Compute penalties
    # 1. Boundary penetration
    boundary_pen = 0.0
    for i in range(N):
        # We can reuse the logic from _boundary_penalty but we already have max_required_R
        # Actually, _boundary_penalty gives the depth. 
        # Let's just sum the positive parts of (proj - apothem)
        # But we need the specific penalty for the optimizer to push them in.
        # The max_required_R already accounts for the worst case vertex.
        # However, if a vertex is outside, max_required_R increases.
        # So the "cost" of being outside is already in max_required_R.
        # We just need to add overlap penalties.
        pass
        
    # 2. Pairwise overlap penalties
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            pen = _sat_penalty(verts_list[i], verts_list[j])
            # Use a smooth penalty or linear?
            # Linear is fine for Nelder-Mead if it's not too steep.
            # Quadratic helps avoid hard stops.
            overlap_pen += pen * pen
            
    # Total cost
    # We want to minimize the side length, but if there are overlaps, we must resolve them first.
    # Weight overlap heavily.
    cost = max_required_R + 10.0 * overlap_pen
    
    return cost

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = float('inf')
    best_params = None
    
    # Define initial layouts
    # Layout 1: Hexagonal lattice
    def get_hex_layout():
        positions = []
        # N=11. 
        # 1 center, 6 ring, 4 from next ring?
        # Or just a 3x4 grid approx?
        # Let's try a compact cluster.
        # Center (0,0)
        # 6 neighbors at distance 2
        # 4 more...
        
        # Simpler: generate on hex grid and pick best 11
        # Or just use a rectangular grid and let optimizer fix it.
        # Let's use a 3x4 grid centered.
        # Spacing 2.0
        dx = 2.0
        dy = 2.0 * math.sqrt(3) / 2.0
        pts = []
        for r in range(-2, 3):
            for c in range(-2, 3):
                x = c * dx + (r % 2) * dx / 2.0
                y = r * dy
                pts.append((x, y))
        # Sort by distance from center
        pts.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = pts[:N]
        
        params = []
        for x, y in selected:
            params.extend([x, y, 0.0])
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)
        
    # Layout 2: Random
    def get_random_layout():
        params = []
        for i in range(N):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            a = random.uniform(0, math.pi)
            params.extend([x, y, a])
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)
        
    # Layout 3: Ring
    def get_ring_layout():
        params = []
        # Center
        params.extend([0.0, 0.0, 0.0])
        # 6 ring
        for i in range(6):
            a = i * math.pi / 3.0
            x = 2.0 * math.cos(a)
            y = 2.0 * math.sin(a)
            params.extend([x, y, 0.0])
        # 4 more from second ring
        for i in range(4):
            a = (i + 0.5) * math.pi / 3.0
            x = 4.0 * math.cos(a)
            y = 4.0 * math.sin(a)
            params.extend([x, y, 0.0])
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)
        
    layouts = [get_hex_layout(), get_ring_layout()]
    for _ in range(5):
        layouts.append(get_random_layout())
        
    for init_params in layouts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_score:
                # Verify validity
                # Extract best params
                p = res.x
                positions = p[:3*N].reshape(N, 3)
                cx, cy = p[3*N:3*N+2]
                outer_angle = p[3*N+2]
                
                # Check overlaps
                valid = True
                verts_list = []
                for i in range(N):
                    x, y, a = positions[i]
                    verts = _hex_vertices(x, y, 1.0, a)
                    verts_list.append(verts)
                    
                for i in range(N):
                    for j in range(i + 1, N):
                        if _sat_penalty(verts_list[i], verts_list[j]) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                    
                if valid:
                    # Calculate true side
                    center = np.array([cx, cy])
                    max_required_R = 0.0
                    for i in range(N):
                        verts = verts_list[i]
                        rel_verts = verts - center
                        for k in range(6):
                            a = outer_angle + k * math.pi / 3.0
                            normal = np.array([math.cos(a), math.sin(a)])
                            proj = np.dot(rel_verts, normal)
                            max_proj = np.max(proj)
                            required_R = max_proj / (math.sqrt(3) / 2.0)
                            if required_R > max_required_R:
                                max_required_R = required_R
                            
                    if max_required_R < 1.0:
                        max_required_R = 1.0
                        
                    if max_required_R < best_score:
                        best_score = max_required_R
                        best_params = p
                        
        except Exception as e:
            pass # Ignore optimization errors
            
    if best_params is not None:
        positions = best_params[:3*N].reshape(N, 3)
        cx, cy = best_params[3*N:3*N+2]
        outer_angle = best_params[3*N+2]
        
        inner = []
        for i in range(N):
            x, y, a = positions[i]
            inner.append([x, y, math.degrees(a)])
            
        return inner, (cx, cy), best_score, math.degrees(outer_angle)
        
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
