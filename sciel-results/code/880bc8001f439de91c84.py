import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin
    
    # Initial configuration: Flower (7) + 4 extensions
    # Axial coords (q, r) -> Cartesian (1.5q, sqrt(3)(r + q/2))
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
        
    coords_A = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    
    # Create initial positions and angles
    init_pos = np.array([axial_to_cartesian(q, r) for q, r in coords_A])
    init_angles = np.zeros(11) # All 0 degrees initially
    
    # Center the cluster
    init_pos -= init_pos.mean(axis=0)
    
    # Variables: 
    # 0-21: x, y for 11 hexagons
    # 22-32: angles for 11 hexagons (radians)
    # 33: outer x
    # 34: outer y
    # 35: outer angle (radians)
    # 36: outer side length (optional, but we can infer it. Let's just optimize positions and outer orientation/pos, and compute side)
    # Actually, to make it a smooth optimization, we can include side in the objective if we penalize containment violation, 
    # OR we can just minimize the required side length for the given configuration.
    # Let's minimize: Required_Side_Length + Penalty_Violations
    
    def objective(params):
        pos = params[:22].reshape(11, 2)
        angles = params[22:33]
        ox, oy = params[33], params[34]
        o_angle = params[35]
        
        # Compute vertices for all inner hexagons
        Vs = []
        for i in range(11):
            V = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.rad2deg(angles[i]))
            Vs.append(V)
            
        # 1. Containment Objective: Minimize the required side length of the enclosing hexagon
        # Normals of the outer hexagon
        o_angle_rad = o_angle
        normals = np.column_stack((np.cos(o_angle_rad + np.deg2rad(60.0) * np.arange(6)), 
                                   np.sin(o_angle_rad + np.deg2rad(60.0) * np.arange(6))))
        
        # Projections of all vertices of all inner hexagons onto outer normals
        # We need the max extent in each normal direction relative to the outer center
        max_proj = np.zeros(6)
        min_proj = np.zeros(6)
        for V in Vs:
            proj = V @ normals.T # (6, 6)
            max_proj = np.maximum(max_proj, proj.max(axis=0))
            min_proj = np.minimum(min_proj, proj.min(axis=0))
            
        # Shift by outer center
        center_proj = np.array([ox, oy]) @ normals.T
        max_proj -= center_proj
        min_proj -= center_proj
        
        # Required apothem is max( max(max_proj), max(-min_proj) )
        # Actually, the hexagon must contain all points.
        # n_k . (p - c) <= apothem
        # n_k . (p - c) >= -apothem
        # So apothem >= max_k ( max_i (n_k . (p_i - c)) )
        # AND apothem >= max_k ( - min_i (n_k . (p_i - c)) )
        # Since the hexagon is symmetric, we just need the max absolute projection?
        # No, the hexagon is defined by 6 half planes.
        # The side length s = 2/sqrt(3) * apothem.
        # We need apothem >= max( max_k(max_proj[k]), max_k(-min_proj[k]) )
        
        required_apothem = np.max(np.maximum(max_proj, -min_proj))
        side = required_apothem * 2.0 / np.sqrt(3.0)
        
        # 2. Separation Penalty
        # We want all pairs to have gap > 1e-6.
        # Penalty = sum( max(0, (1e-6 - gap))^2 )
        # Using a smooth approximation for max(0, x): softplus or just x^2 if x>0.
        # Since gap calculation is non-differentiable at the "switch" of active normal, 
        # we use a heuristic: compute gap using all normals and take the max.
        # This is non-differentiable. 
        # Alternative: Use a repulsive force based on center distance? 
        # No, that's not accurate for hexagons.
        
        # Let's use a simple heuristic penalty: 
        # If centers are too close, push them apart.
        # But we need to respect the hexagon shape.
        
        # A better approach for differentiability:
        # The gap between two convex polygons is the minimum distance between them.
        # This is non-smooth.
        
        # Let's try a different strategy: 
        # Instead of penalty, we can use a "soft" separation constraint in the objective.
        # But L-BFGS-B doesn't handle constraints well.
        
        # Let's stick to minimizing the side length, and use a penalty for overlap.
        # To make it differentiable, we can approximate the overlap volume or just use center distance 
        # with a threshold.
        # Given the complexity, let's just minimize the side length. 
        # If the hexagons overlap, the side length might still be small, but the validity check will fail.
        # We can add a penalty term: sum over pairs of max(0, (dist - min_dist))^2.
        # But min_dist depends on orientation.
        
        # Let's try a "force" approach:
        # For each pair, if they are "close" (center dist < 2), apply a repulsive force.
        # This is a heuristic.
        
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(pos[i] - pos[j])
                # If d < 1.8, they might be overlapping or touching.
                # We want them to be at least some distance apart.
                # For hexagons of side 1, the minimum distance between centers for non-overlap varies with orientation.
                # Max distance is 2 (flat to flat). Min distance is sqrt(3) (point to point).
                # Let's use a soft repulsion for d < 2.0.
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
                    
        return side + 100.0 * penalty
        
    # Initial guess
    x0 = np.concatenate([init_pos.flatten(), init_angles, [0.0, 0.0, 0.0]])
    
    best_val = -np.inf
    best_result = None
    
    # Run optimization
    # We will run multiple trials with different initial perturbations
    
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
            
        if trial == 0:
            x_start = x0
        else:
            # Random perturbation
            x_start = x0.copy()
            x_start[:22] += np.random.normal(0, 0.5, size=(22,))
            x_start[22:33] += np.random.normal(0, 0.2, size=(11,))
            
        res = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
        # Check validity of the current best
        if time.time() > deadline - 5.0:
            break
            
        pos = best_params[:22].reshape(11, 2)
        angles = best_params[22:33]
        ox, oy = best_params[33], best_params[34]
        o_angle = best_params[35]
        
        inner = np.column_stack((pos[:, 0], pos[:, 1], np.rad2deg(angles)))
        outer = np.array([ox, oy, np.rad2deg(o_angle)])
        
        # Compute the actual side length required
        # Re-calculate to be sure
        Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
        o_angle_rad = np.deg2rad(o_angle)
        normals = np.column_stack((np.cos(o_angle_rad + np.deg2rad(60.0) * np.arange(6)), 
                                   np.sin(o_angle_rad + np.deg2rad(60.0) * np.arange(6))))
        max_proj = np.zeros(6)
        min_proj = np.zeros(6)
        for V in Vs:
            proj = V @ normals.T
            max_proj = np.maximum(max_proj, proj.max(axis=0))
            min_proj = np.minimum(min_proj, proj.min(axis=0))
        center_proj = np.array([ox, oy]) @ normals.T
        max_proj -= center_proj
        min_proj -= center_proj
        required_apothem = np.max(np.maximum(max_proj, -min_proj))
        side = required_apothem * 2.0 / np.sqrt(3.0)
        
        if _all_valid(inner, outer, side):
            val = 1.0 / side
            if val > best_val:
                best_val = val
                best_result = (inner, outer, side)
                
    if best_result is None:
        # Fallback
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    return best_result
