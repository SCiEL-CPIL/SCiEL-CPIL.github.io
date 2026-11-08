import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # Try many outer hexagon orientations
    for phi_deg in np.linspace(0, 60, 600):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_result = None
    best_score = float('inf')
    
    def honeycomb_positions(count, pattern):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * np.sqrt(3)
                y = j * 1.5
                positions.append((x, y, 0.0))
        return positions[:count]
    
    # Try several row patterns that sum to 12
    patterns = [
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [4, 4, 4],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
    ]
    
    # Phase 1: Try structured honeycomb patterns
    for pattern in patterns:
        if sum(pattern) != 12:
            continue
        if len(pattern) > 6:
            continue
        
        positions = honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
        
        rot = 30.0
        inner = [(x, y, rot) for (x, y, _) in positions]
        inner_sides = [1.0] * 12
        
        cx, cy, side, angle = _min_enclosing_hexagon(inner, [r[2] for r in inner], inner_sides)
        
        # Add small safety margin for numerical precision
        side_safe = side * 1.0001 + 1e-6
        
        if side_safe < best_score:
            best_score = side_safe
            best_result = (inner, (cx, cy), side_safe, angle)
        
        if time.time() > deadline:
            break
    
    # Phase 2: Use Differential Evolution to optimize positions and rotations
    # This explores the 36D space (12 positions + 12 rotations) globally
    
    def objective_de(params):
        """Objective function for Differential Evolution."""
        if time.time() > deadline:
            return 1e10
        
        xs = params[0:12]
        ys = params[12:24]
        angs = params[24:36]
        
        inner_positions = [(xs[i], ys[i], angs[i]) for i in range(12)]
        inner_sides = [1.0] * 12
        
        cx, cy, side, angle = _min_enclosing_hexagon(inner_positions, angs, inner_sides)
        
        # Check for overlaps using minimum distance between hexagons
        # For simplicity, use center-to-center distance as a proxy
        # Two unit hexagons with same orientation are disjoint if center distance > sqrt(3)
        # With different orientations, this is more complex, so we add a penalty
        for i in range(12):
            for j in range(i+1, 12):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1.7:  # Conservative threshold
                    return 1e10 + (1.7 - d) * 100
        
        return side
    
    # Bounds for DE
    bounds = [(-5, 5)] * 24 + [(-30, 30)] * 12
    
    # Use best honeycomb result as initial guess if available
    if best_result is not None:
        inner_init = best_result[0]
        x0 = []
        for (x, y, a) in inner_init:
            x0.extend([x, y, a])
        x0 = np.array(x0)
    else:
        x0 = np.random.uniform(-3, 3, 36)
    
    # Run DE with limited iterations to stay within time budget
    try:
        if time.time() < deadline - 20:
            result = differential_evolution(
                objective_de, bounds, 
                maxiter=50, popsize=15, tol=1e-3,
                seed=42, init='latinhypercube',
                polish=False, update_interval=5
            )
            if result.fun < best_score:
                xs = result.x[0:12]
                ys = result.x[12:24]
                angs = result.x[24:36]
                inner = [(xs[i], ys[i], angs[i]) for i in range(12)]
                inner_sides = [1.0] * 12
                
                cx, cy, side, angle = _min_enclosing_hexagon(inner, angs, inner_sides)
                side_safe = side * 1.0001 + 1e-6
                
                if side_safe < best_score:
                    best_score = side_safe
                    best_result = (inner, (cx, cy), side_safe, angle)
    except Exception:
        pass
    
    # Phase 3: Local refinement using Nelder-Mead on the best solution
    if best_result is not None and time.time() < deadline - 10:
        inner_init = best_result[0]
        x0 = []
        for (x, y, a) in inner_init:
            x0.extend([x, y, a])
        x0 = np.array(x0)
        
        try:
            res = minimize(objective_de, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                xs = res.x[0:12]
                ys = res.x[12:24]
                angs = res.x[24:36]
                inner = [(xs[i], ys[i], angs[i]) for i in range(12)]
                inner_sides = [1.0] * 12
                
                cx, cy, side, angle = _min_enclosing_hexagon(inner, angs, inner_sides)
                side_safe = side * 1.0001 + 1e-6
                
                if side_safe < best_score:
                    best_score = side_safe
                    best_result = (inner, (cx, cy), side_safe, angle)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
