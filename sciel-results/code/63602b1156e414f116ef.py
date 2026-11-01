"""Improved program using smooth surrogate + L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

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

def smooth_min(values, beta=50.0):
    """Soft-min using log-sum-exp (negative log-sum-exp of negated values)."""
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m < -100:
        return m
    shifted = values - m
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return m - np.log(np.sum(exp_vals)) / beta

def smooth_max(values, beta=50.0):
    """Soft-max using log-sum-exp."""
    return -smooth_min(-values, beta)

def compute_clearances_fast(centers, outer_center, outer_angle, outer_side):
    """
    Compute all clearance values:
    - For each pair of inner hexagons: min separation (positive = separated)
    - For each inner hexagon: min distance to outer boundary (positive = inside)
    Returns (min_pair_sep, min_outer_clearance, all_clearances)
    """
    n = len(centers)
    pair_clearances = []
    outer_clearances = []
    
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    apothem = outer_side * np.sqrt(3) / 2
    
    # Precompute vertices for all inner hexagons
    inner_verts = []
    for c in centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        inner_verts.append(v)
    
    # Pairwise separations using SAT with all 12 axes
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            # Collect all edge normals
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k + 1) % 6] - v[k]
                    elen = np.linalg.norm(edge)
                    if elen > 1e-12:
                        nrm = np.array([-edge[1], edge[0]]) / elen
                        axes.append(nrm)
            
            min_sep = float('inf')
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if sep < min_sep:
                    min_sep = sep
            
            pair_clearances.append(min_sep)
    
    # Outer clearance: for each inner vertex, compute signed distance to outer boundary
    for i in range(n):
        v = inner_verts[i]
        for vert in v:
            rel = np.array([vert[0] - outer_center[0], vert[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            # Hexagon constraint: max over 6 directions of |dot|
            max_proj = 0.0
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                if d > max_proj:
                    max_proj = d
            
            clearance = apothem - max_proj
            outer_clearances.append(clearance)
    
    min_pair = min(pair_clearances) if pair_clearances else 0.0
    min_outer = min(outer_clearances) if outer_clearances else 0.0
    
    return min_pair, min_outer, pair_clearances, outer_clearances

def objective_smooth(params, outer_center, outer_angle, outer_side, beta=30.0):
    """
    Smooth objective: penalize negative clearances using soft-min.
    params: flat array of [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns scalar objective value (lower is better, should be <= 0 for valid).
    """
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append([x, y, a])
    
    min_pair, min_outer, pair_cl, outer_cl = compute_clearances_fast(
        centers, outer_center, outer_angle, outer_side
    )
    
    # Soft-min of all clearances (we want all to be >= 0)
    all_cl = pair_cl + outer_cl
    if len(all_cl) == 0:
        return 0.0
    
    # Penalize negative values: use soft-min which is smooth
    # If all clearances >= 0, soft-min >= 0, objective = 0
    # If some are negative, soft-min < 0, objective = soft-min (negative penalty)
    sm = smooth_min(all_cl, beta)
    
    # We want to maximize the minimum clearance, so minimize -sm
    # But we also want to minimize outer_side, which is fixed here
    # So objective = -sm (we want sm to be as large as possible, i.e., close to 0 or positive)
    # Actually, for feasibility we just need sm >= 0. Let's use a penalty.
    
    return -sm

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
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_pair, _, _, _ = compute_clearances_fast(
                [centers[i], centers[j]], (0,0), 0, 100
            )
            if min_pair < -1e-9:
                return False
    return True

def sat_min_separation(c1, c2):
    """Compute minimum separation between two hexagons using SAT."""
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
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
    
    return min_sep

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
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

def initial_config_staggered():
    """Staggered grid arrangement."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    # 4x3 grid with stagger
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            x -= 1.5 * dx
            y -= dy
            centers.append([x, y, 0.0])
    return centers

def smooth_objective_compact(params, outer_center, outer_angle, beta=30.0):
    """
    Objective that minimizes the 'smooth' outer side length and overlaps.
    This encourages compactness and validity.
    """
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append([x, y, a])
    
    # Calculate smooth outer side
    # We approximate the required outer side by the smooth max of the extents
    # Extent of a vertex v relative to outer center/angle
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    extents = []
    for c in centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        for vert in v:
            rel = np.array([vert[0] - outer_center[0], vert[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            max_proj = 0.0
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                if d > max_proj:
                    max_proj = d
            extents.append(max_proj)
            
    # Smooth max of extents gives a smooth approximation of the required apothem
    smooth_apothem = smooth_max(extents, beta)
    smooth_side = smooth_apothem / (np.sqrt(3) / 2)
    
    # Calculate overlaps
    min_pair, min_outer, pair_cl, outer_cl = compute_clearances_fast(
        centers, outer_center, outer_angle, smooth_side
    )
    all_cl = pair_cl + outer_cl
    if len(all_cl) == 0:
        return smooth_side
    
    # Penalty for negative clearances (overlaps/outside)
    # We want all_cl >= 0. If not, penalize.
    # Use smooth max of negative clearances
    neg_cl = [-c for c in all_cl if c < 0]
    if neg_cl:
        penalty = smooth_max(neg_cl, beta) ** 2 * 100.0
    else:
        penalty = 0.0
        
    return smooth_side + penalty

def lbfghs_polish(centers, outer_center, outer_angle, outer_side, time_limit=20.0):
    """
    Polish a configuration using L-BFGS-B with the smooth objective.
    Optimizes inner hexagon positions and angles.
    """
    start_time = time.time()
    
    # Flatten parameters
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        return smooth_objective_compact(p, outer_center, outer_angle)
    
    def grad(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        eps = 1e-5
        g = np.zeros_like(p)
        f0 = obj(p)
        for i in range(len(p)):
            p_p = p.copy()
            p_p[i] += eps
            fp = obj(p_p)
            g[i] = (fp - f0) / eps
        return g
    
    result = minimize(
        obj, params, jac=grad, method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("staggered", initial_config_staggered()),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Use L-BFGS-B directly with the compacting objective
        # This is faster and more directed than SA for this specific objective
        polish_time = min(30.0, DEADLINE - elapsed - 5)
        
        try:
            polished = lbfghs_polish(config, (0.0, 0.0), 0.0, 10.0, time_limit=polish_time)
        except Exception:
            polished = config
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Compute final required side
        final_side = required_outer_side(polished, (0.0, 0.0), 0.0)
        
        # Validate
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(polished[i], polished[j])
                if s < -1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and final_side < best_score:
            best_score = final_side
            best_result = (polished, (0.0, 0.0), final_side, 0.0)
            
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
