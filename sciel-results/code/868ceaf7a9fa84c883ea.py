<<<<<<< SEARCH
"""Improved program using simulated annealing with force-based model."""
import numpy as np
import time

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    Also returns the axis of minimum separation.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check containment
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def energy(centers, outer_center, outer_angle):
    """
    Compute energy of configuration.
    Returns (required_side, overlap_penalty, forces).
    forces is a list of (fx, fy, fa) for each inner hexagon.
    """
    n = len(centers)
    forces = [(0.0, 0.0, 0.0)] * n
    
    # Compute required outer side
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    # Compute overlap forces using SAT
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep, axis = sat_min_separation(centers[i], centers[j])
            if sep < 0:
                # Overlapping: push apart along axis
                overlap_penalty += (-sep) ** 2
                push = -sep * 2.0  # force magnitude
                if axis is not None:
                    # i should move in -axis direction, j in +axis direction
                    fi = (forces[i][0] - axis[0] * push, 
                          forces[i][1] - axis[1] * push, 
                          forces[i][2])
                    fj = (forces[j][0] + axis[0] * push, 
                          forces[j][1] + axis[1] * push, 
                          forces[j][2])
                    forces[i] = fi
                    forces[j] = fj
    
    # Compute containment forces
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for i in range(n):
        px, py, pa = centers[i]
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        fx = 0.0
        fy = 0.0
        fa = 0.0
        
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = rx * nx + ry * ny
                limit = req_side * np.sqrt(3) / 2
                
                if d > limit:
                    # Push back toward center
                    excess = d - limit
                    # Force in direction opposite to normal, rotated back
                    fx_local = -nx * excess
                    fy_local = -ny * excess
                    # Rotate back to world coordinates
                    fx += fx_local * np.cos(oa_rad) - fy_local * np.sin(oa_rad)
                    fy += fx_local * np.sin(oa_rad) + fy_local * np.cos(oa_rad)
        
        forces[i] = (forces[i][0] + fx, forces[i][1] + fy, forces[i][2] + fa)
    
    return req_side, overlap_penalty, forces

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    # Hexagonal lattice spacing: 2*sqrt(3) for non-overlapping unit hexagons
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    # Center at origin, arrange in hexagonal pattern
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def simulated_annealing(init_centers, max_iter=50000, T_start=2.0, T_end=0.01, 
                        cooling_rate=0.99995, time_limit=140.0):
    """
    Run simulated annealing to optimize the packing.
    Returns (best_centers, best_outer_center, best_outer_side, best_outer_angle).
    """
    start_time = time.time()
    n = len(init_centers)
    
    # Initial state
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial energy
    req_side, overlap_pen, forces = energy(centers, outer_center, outer_angle)
    current_score = req_side + 10.0 * overlap_pen
    
    best_centers = [list(c) for c in centers]
    best_outer_center = list(outer_center)
    best_outer_angle = outer_angle
    best_score = current_score
    
    T = T_start
    
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Generate perturbation
        new_centers = [list(c) for c in centers]
        new_outer_center = list(outer_center)
        new_outer_angle = outer_angle
        
        # Perturb a random inner hexagon
        i = np.random.randint(n)
        step_size = T * 0.5
        new_centers[i][0] += np.random.normal(0, step_size)
        new_centers[i][1] += np.random.normal(0, step_size)
        new_centers[i][2] += np.random.normal(0, step_size * 30)
        new_centers[i][2] = new_centers[i][2] % 60
        
        # Occasionally perturb outer parameters
        if np.random.random() < 0.1:
            new_outer_center[0] += np.random.normal(0, step_size * 0.3)
            new_outer_center[1] += np.random.normal(0, step_size * 0.3)
        if np.random.random() < 0.05:
            new_outer_angle += np.random.normal(0, step_size * 10)
            new_outer_angle = new_outer_angle % 360
        
        # Compute new energy
        new_req_side, new_overlap_pen, new_forces = energy(new_centers, new_outer_center, new_outer_angle)
        new_score = new_req_side + 10.0 * new_overlap_pen
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            centers = new_centers
            outer_center = new_outer_center
            outer_angle = new_outer_angle
            current_score = new_score
            req_side, overlap_pen, forces = new_req_side, new_overlap_pen, new_forces
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_centers = [list(c) for c in centers]
                best_outer_center = list(outer_center)
                best_outer_angle = outer_angle
        else:
            # Use forces to guide next move (optional refinement)
            pass
        
        # Cool down
        T = max(T_end, T * cooling_rate)
        
        # Periodically recenter to prevent drift
        if it % 1000 == 0 and it > 0:
            cx = np.mean([c[0] for c in centers])
            cy = np.mean([c[1] for c in centers])
            for c in centers:
                c[0] -= cx
                c[1] -= cy
            outer_center[0] -= cx
            outer_center[1] -= cy
    
    # Final validation and cleanup
    final_req_side = required_outer_side(best_centers, best_outer_center, best_outer_angle)
    
    # Verify validity
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to fix by slightly expanding
        return None
    
    return (best_centers, best_outer_center, final_req_side, best_outer_angle)

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Multiple initial configurations
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999))
    ]
    
    for name, config in configs:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        try:
            result = simulated_annealing(
                config, 
                max_iter=30000,
                T_start=2.0,
                T_end=0.005,
                cooling_rate=0.99997,
                time_limit=DEADLINE - (time.time() - start_time) - 3
            )
            
            if result is not None:
                centers, outer_center, outer_side, outer_angle = result
                
                # Final validation
                if check_validity(centers, tuple(outer_center), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (centers, tuple(outer_center), outer_side, outer_angle)
        except Exception as e:
            pass
    
    # Fallback: return best found or a valid default
    if best_result is not None:
        return best_result
    
    # Default valid packing (larger outer hexagon)
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    # Compute required side for default
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
=======
"""Improved program using smooth surrogate + L-BFGS-B polish on top of SA."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = np.sqrt(3.0)

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(thetas),
        cy + side * np.sin(thetas)
    ])

def hexagon_normals(center, side, angle_deg=0.0):
    """Return 6 outward normals of a regular hexagon (unit vectors)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6)) + np.radians(30)
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def sat_min_separation_fast(c1, c2):
    """
    Fast SAT minimum separation between two unit hexagons.
    Returns (min_sep, best_axis).
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    # Get all 12 edge normals (6 from each hexagon)
    n1 = hexagon_normals((0, 0), 1.0, c1[2])
    n2 = hexagon_normals((0, 0), 1.0, c2[2])
    axes = np.vstack([n1, n2])
    
    # Project all vertices onto all axes
    proj1 = v1 @ axes.T  # (6, 12)
    proj2 = v2 @ axes.T  # (6, 12)
    
    # Separation for each axis
    sep1 = proj1.min(axis=0) - proj2.max(axis=0)
    sep2 = proj2.min(axis=0) - proj1.max(axis=0)
    seps = np.maximum(sep1, sep2)
    
    min_idx = np.argmin(seps)
    min_sep = seps[min_idx]
    best_axis = axes[min_idx]
    
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oa_rad = np.radians(outer_angle)
    R = np.array([[np.cos(-oa_rad), -np.sin(-oa_rad)],
                  [np.sin(-oa_rad), np.cos(-oa_rad)]])
    
    # Outer hexagon support function: for each of 6 directions
    outer_normals = hexagon_normals((0, 0), 1.0, 0.0)
    
    max_extent = 0.0
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        rel = inner_verts - np.array(outer_center)
        rel_rot = rel @ R.T  # (6, 2)
        
        for k in range(6):
            d = np.abs(rel_rot @ outer_normals[k])
            max_extent = max(max_extent, d)
    
    return max_extent / (SQRT3 / 2)

def smooth_required_outer_side(centers, outer_center, outer_angle, beta=50.0):
    """Smooth version using log-sum-exp for the max."""
    oa_rad = np.radians(outer_angle)
    R = np.array([[np.cos(-oa_rad), -np.sin(-oa_rad)],
                  [np.sin(-oa_rad), np.cos(-oa_rad)]])
    
    outer_normals = hexagon_normals((0, 0), 1.0, 0.0)
    
    all_extents = []
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        rel = inner_verts - np.array(outer_center)
        rel_rot = rel @ R.T
        
        for k in range(6):
            d = np.abs(rel_rot @ outer_normals[k])
            all_extents.append(d)
    
    all_extents = np.array(all_extents)
    # Log-sum-exp for smooth max
    m = np.max(all_extents)
    if m < 1e-12:
        return 0.0
    smooth_max = m + np.log(np.mean(np.exp((all_extents - m) / beta))) * beta
    return smooth_max / (SQRT3 / 2)

def smooth_min_separation(c1, c2, beta=30.0):
    """Smooth minimum separation using log-sum-exp."""
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    n1 = hexagon_normals((0, 0), 1.0, c1[2])
    n2 = hexagon_normals((0, 0), 1.0, c2[2])
    axes = np.vstack([n1, n2])
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep1 = proj1.min(axis=0) - proj2.max(axis=0)
    sep2 = proj2.min(axis=0) - proj1.max(axis=0)
    seps = np.maximum(sep1, sep2)
    
    # Smooth min
    m = np.min(seps)
    if m > 1e-12:
        return m
    smooth_min = m - np.log(np.mean(np.exp(-(seps - m) / beta))) * beta
    return smooth_min

def smooth_objective(params, beta=30.0):
    """
    Smooth objective for optimization.
    params: flat array [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, ox,oy, oangle]
    Returns objective value (outer_side + penalty for overlaps).
    """
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append((x, y, a % 60.0))
    
    outer_center = (params[3*n], params[3*n+1])
    outer_angle = params[3*n+2]
    
    # Required outer side
    req_side = smooth_required_outer_side(centers, outer_center, outer_angle, beta=beta)
    
    # Overlap penalty using smooth min separation
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = smooth_min_separation(centers[i], centers[j], beta=beta)
            if sep < 0:
                overlap_penalty += sep * sep
    
    return req_side + 10.0 * overlap_penalty

def objective_with_side(params, beta=30.0):
    """Objective that returns both the side length and overlap penalty."""
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append((x, y, a % 60.0))
    
    outer_center = (params[3*n], params[3*n+1])
    outer_angle = params[3*n+2]
    
    req_side = smooth_required_outer_side(centers, outer_center, outer_angle, beta=beta)
    
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = smooth_min_separation(centers[i], centers[j], beta=beta)
            if sep < 0:
                overlap_penalty += sep * sep
    
    return req_side + 10.0 * overlap_penalty

def lbfghs_polish(centers, outer_center, outer_angle, time_limit=5.0):
    """
    Polish a configuration using L-BFGS-B with smooth surrogate.
    Returns (polished_centers, polished_outer_center, polished_outer_angle, polished_side).
    """
    start_time = time.time()
    n = N
    
    # Flatten parameters
    params = np.zeros(3 * n + 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2] % 60.0
    params[3*n] = outer_center[0]
    params[3*n+1] = outer_center[1]
    params[3*n+2] = outer_angle
    
    # Use decreasing beta for more precise optimization
    best_params = params.copy()
    best_val = objective_with_side(params, beta=30.0)
    
    for beta in [30.0, 15.0, 7.0, 3.0]:
        if time.time() - start_time > time_limit:
            break
        
        def obj(p):
            return objective_with_side(p, beta=beta)
        
        try:
            res = minimize(obj, best_params, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxls': 20})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract polished result
    polished_centers = []
    for i in range(n):
        x, y, a = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        polished_centers.append([x, y, a % 60.0])
    
    polished_outer_center = (best_params[3*n], best_params[3*n+1])
    polished_outer_angle = best_params[3*n+2]
    
    # Compute exact required side
    polished_side = required_outer_side(polished_centers, polished_outer_center, polished_outer_angle)
    
    return polished_centers, polished_outer_center, polished_outer_angle, polished_side

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check containment
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation_fast(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * SQRT3
    dy = 3.0
    
    # Hexagonal pattern: 1 center + 6 ring + 5 more to make 12
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*SQRT3/2), (-dx/2, dy*SQRT3/2),
        (dx/2, -dy*SQRT3/2), (-dx/2, -dy*SQRT3/2),
        (3*dx/2, dy*SQRT3/2), (-3*dx/2, dy*SQRT3/2),
        (3*dx/2, -dy*SQRT3/2), (-3*dx/2, -dy*SQRT3/2),
        (0, dy*SQRT3), (0, -dy*SQRT3)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_staggered():
    """Create a staggered grid arrangement."""
    centers = []
    dx = 2.0 * SQRT3
    dy = 3.0
    
    # 3 rows of 4, staggered
    for row in range(3):
        y = (row - 1) * dy * SQRT3 / 2
        offset = 0 if row % 2 == 0 else dx / 2
        for col in range(4):
            x = (col - 1.5) * dx + offset
            centers.append([x, y, 0.0])
    
    return centers[:N]

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def simulated_annealing(init_centers, max_iter=20000, T_start=1.5, T_end=0.01, 
                        cooling_rate=0.99995, time_limit=20.0):
    """
    Run simulated annealing with smooth surrogate.
    Returns (best_centers, best_outer_center, best_outer_side, best_outer_angle).
    """
    start_time = time.time()
    n = len(init_centers)
    
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial score
    current_score = objective_with_side(_flatten(centers, outer_center, outer_angle), beta=30.0)
    
    best_centers = [list(c) for c in centers]
    best_outer_center = list(outer_center)
    best_outer_angle = outer_angle
    best_score = current_score
    
    T = T_start
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_centers = [list(c) for c in centers]
        new_outer_center = list(outer_center)
        new_outer_angle = outer_angle
        
        i = np.random.randint(n)
        step_size = T * 0.3
        new_centers[i][0] += np.random.normal(0, step_size)
        new_centers[i][1] += np.random.normal(0, step_size)
        new_centers[i][2] += np.random.normal(0, step_size * 20)
        new_centers[i][2] = new_centers[i][2] % 60
        
        if np.random.random() < 0.08:
            new_outer_center[0] += np.random.normal(0, step_size * 0.2)
            new_outer_center[1] += np.random.normal(0, step_size * 0.2)
        if np.random.random() < 0.04:
            new_outer_angle += np.random.normal(0, step_size * 8)
            new_outer_angle = new_outer_angle % 360
        
        new_score = objective_with_side(_flatten(new_centers, new_outer_center, new_outer_angle), beta=30.0)
        
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-6)):
            centers = new_centers
            outer_center = new_outer_center
            outer_angle = new_outer_angle
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_centers = [list(c) for c in centers]
                best_outer_center = list(outer_center)
                best_outer_angle = outer_angle
        
        T = max(T_end, T * cooling_rate)
        
        if it % 500 == 0 and it > 0:
            cx = np.mean([c[0] for c in centers])
            cy = np.mean([c[1] for c in centers])
            for c in centers:
                c[0] -= cx
                c[1] -= cy
            outer_center[0] -= cx
            outer_center[1] -= cy
    
    final_req_side = required_outer_side(best_centers, best_outer_center, best_outer_angle)
    
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation_fast(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return (best_centers, best_outer_center, final_req_side, best_outer_angle)

def _flatten(centers, outer_center, outer_angle):
    """Flatten centers and outer params into a 1D array."""
    n = len(centers)
    params = np.zeros(3 * n + 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2] % 60.0
    params[3*n] = outer_center[0]
    params[3*n+1] = outer_center[1]
    params[3*n+2] = outer_angle
    return params

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("staggered", initial_config_staggered()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999)),
        ("random4", initial_config_random(777)),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 8:
            break
        
        sa_time = min(15.0, DEADLINE - elapsed - 5)
        polish_time = min(5.0, DEADLINE - elapsed - sa_time - 3)
        
        if sa_time < 3.0:
            break
        
        try:
            result = simulated_annealing(
                config,
                max_iter=15000,
                T_start=1.5,
                T_end=0.01,
                cooling_rate=0.99995,
                time_limit=sa_time
            )
            
            if result is not None:
                centers, outer_center, outer_side, outer_angle = result
                
                # Polish with L-BFGS-B
                elapsed2 = time.time() - start_time
                polish_time2 = min(5.0, DEADLINE - elapsed2 - 3)
                if polish_time2 > 1.0:
                    p_centers, p_outer_center, p_outer_angle, p_side = lbfghs_polish(
                        centers, outer_center, outer_angle, time_limit=polish_time2
                    )
                    if check_validity(p_centers, p_outer_center, p_side, p_outer_angle):
                        if p_side < best_score:
                            best_score = p_side
                            best_result = (p_centers, p_outer_center, p_side, p_outer_angle)
                
                # Also check unpolished
                if check_validity(centers, tuple(outer_center), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (centers, tuple(outer_center), outer_side, outer_angle)
        except Exception as e:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
>>>>>>> REPLACE
