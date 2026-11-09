"""Packing 12 unit hexagons in minimal regular hexagon using L-BFGS-B optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side length."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([[cx + side * np.cos(angle_rad + k * np.pi / 3.0),
                       cy + side * np.sin(angle_rad + k * np.pi / 3.0)] for k in range(6)])
    return verts

def hex_inradius(side=1.0):
    return side * np.sqrt(3) / 2.0

def hex_support(vertex_array, direction):
    """Compute support function of a hexagon (array of 6 vertices) in a given direction."""
    return np.max(vertex_array @ direction)

def min_distance_between_hexagons(v1, v2):
    """
    Compute minimum distance between two convex hexagons given their vertices.
    Returns 0 if overlapping, positive if separated.
    Uses the separating axis theorem: check all 12 edge normals.
    """
    # Get edge normals for both hexagons
    normals = []
    for v in [v1, v2]:
        for k in range(6):
            e = v[(k+1) % 6] - v[k]
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-12:
                n = n / n_norm
                normals.append(n)
    
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    d = c2 - c1
    
    min_dist = float('inf')
    for n in normals:
        # Project both hexagons onto normal direction
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        # The gap in this direction
        gap = (np.min(proj2) - np.max(proj1))
        if gap < min_dist:
            min_dist = gap
    
    return min_dist

def outer_hex_constraint_violation(px, py, L, angle_outer_deg=0.0):
    """
    Compute violation of point (px,py) being inside outer hexagon of side L centered at origin.
    Returns 0 if inside, positive if outside.
    """
    angle_rad = np.deg2rad(angle_outer_deg)
    # Rotate point to outer hexagon frame
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = cos_a * px - sin_a * py
    ly = sin_a * px + cos_a * py
    
    inradius = hex_inradius(L)
    
    # Three normal directions for flat-top hexagon
    d0 = np.array([1.0, 0.0])
    d1 = np.array([0.5, np.sqrt(3)/2.0])
    d2 = np.array([-0.5, np.sqrt(3)/2.0])
    p = np.array([lx, ly])
    
    h0 = abs(np.dot(p, d0))
    h1 = abs(np.dot(p, d1))
    h2 = abs(np.dot(p, d2))
    
    # Return max violation (smooth approximation using soft-max)
    violations = np.array([h0 - inradius, h1 - inradius, h2 - inradius])
    # Use soft-max for smoothness
    beta = 50.0
    max_v = np.max(violations)
    if max_v <= 0:
        return 0.0
    log_sum_exp = max_v + np.log(np.sum(np.exp((violations - max_v) * beta)) / 3.0) / beta
    return log_sum_exp

def objective(params, L):
    """
    params: 3*N array of [x1,y1,a1, ..., xN,y2,yN,aN]
    L: outer hexagon side length
    Returns: smooth penalty function
    """
    penalty = 0.0
    
    # Precompute all hexagon vertices
    hex_verts = []
    for i in range(N):
        xi = params[3*i]
        yi = params[3*i+1]
        ai = params[3*i+2]
        vi = hex_vertices(xi, yi, ai, 1.0)
        hex_verts.append(vi)
    
    # Check all vertices of inner hexagons are inside outer hexagon
    for i in range(N):
        vi = hex_verts[i]
        for vx, vy in vi:
            viol = outer_hex_constraint_violation(vx, vy, L, 0.0)
            penalty += viol ** 2
    
    # Check non-overlap between all pairs using smooth penalty
    for i in range(N):
        for j in range(i+1, N):
            dist = min_distance_between_hexagons(hex_verts[i], hex_verts[j])
            if dist < 0:
                penalty += dist ** 2
    
    return penalty

def full_objective(params):
    """
    params: 3*N + 1 array = [x1,y1,a1, ..., xN,yN,aN, L]
    """
    L = params[-1]
    inner_params = params[:-1]
    return L + 50.0 * objective(inner_params, L)

def gradient(params):
    """Numerical gradient for full_objective."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = full_objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = full_objective(params_plus)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110 second limit (safety margin)
    
    best_L = 40.0
    best_params = None
    
    # Initial guesses: various configurations
    initial_guesses = []
    
    # 1. Hexagonal grid arrangement - 4-4-4 rows
    for spacing_x in [2.0, 2.2, 2.5]:
        for spacing_y in [1.7, 1.8, 2.0]:
            params = np.zeros(3*N + 1)
            for i in range(N):
                row = i // 4
                col = i % 4
                x_offset = (col - 1.5) * spacing_x
                y_offset = (row - 1.5) * spacing_y
                # Stagger alternate rows
                if row % 2 == 1:
                    x_offset += spacing_x * 0.5
                params[3*i] = x_offset
                params[3*i+1] = y_offset
                params[3*i+2] = 0.0
            params[-1] = 8.0
            initial_guesses.append(params)
    
    # 2. Ring arrangement
    for radius in [2.5, 3.0, 3.5]:
        params = np.zeros(3*N + 1)
        for i in range(N):
            angle = 2 * np.pi * i / N
            params[3*i] = radius * np.cos(angle)
            params[3*i+1] = radius * np.sin(angle)
            params[3*i+2] = 0.0
        params[-1] = 7.0
        initial_guesses.append(params)
    
    # 3. Two rings
    for r1 in [1.5, 2.0]:
        for r2 in [3.5, 4.0]:
            params = np.zeros(3*N + 1)
            # Inner ring: 6 hexagons
            for i in range(6):
                angle = 2 * np.pi * i / 6
                params[3*i] = r1 * np.cos(angle)
                params[3*i+1] = r1 * np.sin(angle)
                params[3*i+2] = 0.0
            # Outer ring: 6 hexagons
            for i in range(6):
                angle = 2 * np.pi * (i + 0.5) / 6
                params[3*(6+i)] = r2 * np.cos(angle)
                params[3*(6+i)+1] = r2 * np.sin(angle)
                params[3*(6+i)+2] = 0.0
            params[-1] = 7.0
            initial_guesses.append(params)
    
    # 4. Random configurations
    np.random.seed(42)
    for _ in range(8):
        params = np.zeros(3*N + 1)
        for i in range(N):
            params[3*i] = np.random.uniform(-4, 4)
            params[3*i+1] = np.random.uniform(-4, 4)
            params[3*i+2] = np.random.uniform(0, 60)
        params[-1] = 9.0
        initial_guesses.append(params)
    
    # 5. Linear arrangement
    params = np.zeros(3*N + 1)
    for i in range(N):
        params[3*i] = (i - 5.5) * 2.1
        params[3*i+1] = 0.0
        params[3*i+2] = 0.0
    params[-1] = 15.0
    initial_guesses.append(params)
    
    # 6. 2x6 grid
    for sx in [2.0, 2.2]:
        for sy in [1.7, 1.9]:
            params = np.zeros(3*N + 1)
            for i in range(N):
                row = i // 6
                col = i % 6
                params[3*i] = (col - 2.5) * sx
                params[3*i+1] = (row - 0.5) * sy
                params[3*i+2] = 0.0
            params[-1] = 10.0
            initial_guesses.append(params)
    
    # Optimize from each initial guess
    for idx, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Use L-BFGS-B with numerical gradient
            result = minimize(
                full_objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            L_candidate = result.x[-1]
            inner_params = result.x[:-1]
            penalty = objective(inner_params, L_candidate)
            
            if penalty < 0.05:  # Valid solution
                if L_candidate < best_L:
                    best_L = L_candidate
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time permits
    if best_params is not None and time.time() < deadline - 15.0:
        try:
            result = minimize(
                full_objective,
                best_params,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            L_candidate = result.x[-1]
            inner_params = result.x[:-1]
            penalty = objective(inner_params, L_candidate)
            if penalty < 0.05 and L_candidate < best_L:
                best_L = L_candidate
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final answer
    if best_params is not None:
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner.append([x, y, a])
        center = (0.0, 0.0)
        side = best_L
        angle = 0.0
    else:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
