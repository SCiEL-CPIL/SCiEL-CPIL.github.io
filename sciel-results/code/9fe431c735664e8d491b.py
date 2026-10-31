import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_symmetric_inits():
    """
    Generate initial configurations based on high-symmetry patterns.
    We assume the optimal configuration has significant symmetry.
    """
    inits = []
    center = np.array([1/3, 1/3])
    
    # Pattern 1: Center + Hexagon + 4 "Corners"
    # The 4 corners are likely near the vertices of the triangle.
    # We parameterize the hexagon radius and the corner offset.
    for r_hex in [0.1, 0.15, 0.2, 0.25]:
        for corner_off in [0.1, 0.15, 0.2]:
            pts = [center.copy()]
            
            # 6 points in a hexagon
            for k in range(6):
                theta = np.pi/3 * k
                # Approximate barycentric position for a point at (r, theta) from center
                # Center is (0.5, H/3) in Cartesian? No, (0.5, H/3) is centroid.
                # Centroid in bary is (1/3, 1/3).
                # Cartesian offset: dx = r cos(theta), dy = r sin(theta)
                # Barycentric offset: du = -dx/2 + dy/H, dv = dx/2 + dy/H
                dx = r_hex * np.cos(theta)
                dy = r_hex * np.sin(theta)
                du = -dx / 2 + dy / H
                dv = dx / 2 + dy / H
                pt = center + np.array([du, dv])
                if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                    pts.append(pt.copy())
            
            # 4 points near vertices
            # Vertex 1 (0,0) in bary is (0,0). Shift inwards.
            # Vertex 2 (1,0) in bary is (1,0). Shift inwards.
            # Vertex 3 (0.5, H) in bary is (0,1). Shift inwards.
            
            # Shift towards center
            # V1_bary = (0,0) -> (corner_off, corner_off)
            pts.append(np.array([corner_off, corner_off]))
            # V2_bary = (1,0) -> (1-corner_off, corner_off)
            pts.append(np.array([1-corner_off, corner_off]))
            # V3_bary = (0,1) -> (corner_off, 1-corner_off)
            pts.append(np.array([corner_off, 1-corner_off]))
            
            # We have 1 + 6 + 3 = 10 points. We need 11.
            # Add one more point. Maybe on an axis?
            # Try placing it on the vertical axis of symmetry (u=1/3? No, u=1/3, v=1/3 is center).
            # The axis of symmetry for V3 is u=1/3? No.
            # Symmetry axes pass through vertices and midpoints of opposite sides.
            # Axis through V3 (0,1) and midpoint of V1-V2 (0.5, 0).
            # In bary: (0,1) to (0.5, 0). Line: u = 0.5 * (1-v)?
            # Let's just add a point slightly off-center or on an axis.
            # Try u=0.5, v=0.1
            pts.append(np.array([0.5, 0.1]))
            
            if len(pts) == N:
                inits.append(np.array(pts))
                
    # Pattern 2: 3 near vertices, 3 near midpoints, 1 center, 4 distributed
    for v_off in [0.05, 0.1, 0.15]:
        pts = []
        # 3 near vertices
        pts.append(np.array([v_off, v_off]))
        pts.append(np.array([1-v_off, v_off]))
        pts.append(np.array([v_off, 1-v_off]))
        
        # 3 near midpoints
        # Midpoint V1-V2: (0.5, 0) in bary? No, V1=(1,0), V2=(0,0)?
        # V0=(0,0), V1=(1,0), V2=(0,1).
        # Mid V0-V1: (0.5, 0). Shift in: (0.5-v_off, v_off)?
        pts.append(np.array([0.5 - v_off/2, v_off]))
        # Mid V1-V2: (0.5, 0.5). Shift in: (0.5-v_off, 0.5-v_off)?
        pts.append(np.array([0.5 - v_off, 0.5 - v_off]))
        # Mid V0-V2: (0, 0.5). Shift in: (v_off, 0.5 - v_off/2)?
        pts.append(np.array([v_off, 0.5 - v_off/2]))
        
        # Center
        pts.append(np.array([1/3, 1/3]))
        
        # 4 more. Try a small square or diamond around center?
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = 0.2
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            du = -dx / 2 + dy / H
            dv = dx / 2 + dy / H
            pt = np.array([1/3, 1/3]) + np.array([du, dv])
            if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                pts.append(pt)
        
        if len(pts) == N:
            inits.append(np.array(pts))

    # Pattern 3: Grid-like
    for scale in [0.2, 0.25, 0.3]:
        pts = []
        # Generate points on a triangular grid
        for i in range(int(1/scale) + 1):
            for j in range(int((1 - i*scale)/scale) + 1):
                u = i * scale
                v = j * scale
                if u + v < 1 - scale/2:
                    pts.append([u, v])
        
        if len(pts) >= N:
            # Pick N points that are most spread out
            # Simple heuristic: just take the first N
            inits.append(np.array(pts[:N]))

    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [10.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.8:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            if result2.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result2.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if result3.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result3.x.reshape(N, 2))
        except Exception:
            pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Symmetric starts
    inits.extend(generate_symmetric_inits())
    
    # 2. Random starts (fewer, as symmetric ones are stronger)
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 3. Random starts on symmetry axes
    for i in range(10):
        pts = []
        for j in range(N):
            # Choose a random axis and position
            axis = rng.integers(0, 3)
            t = rng.random()
            if axis == 0: # u = v
                u = t * 0.5
                v = t * 0.5
            elif axis == 1: # u + v = 1
                u = t
                v = 1 - t
            else: # v = 0
                u = t
                v = 0
            pts.append([u, v])
        inits.append(np.array(pts))

    # Deduplicate inits
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.05):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Sort inits by initial objective (best first)
    inits_with_val = []
    for init in inits:
        val = -hard_min_obj(init.flatten())
        inits_with_val.append((val, init))
    inits_with_val.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates
    for i, (val, init) in enumerate(inits_with_val):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on rank
        if i < 5:
            time_per_init = 8.0
        elif i < 15:
            time_per_init = 4.0
        else:
            time_per_init = 2.0
            
        if time.time() + time_per_init > deadline - 5:
            break
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-16}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
