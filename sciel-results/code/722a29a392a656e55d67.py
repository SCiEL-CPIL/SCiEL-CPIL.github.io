"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses free-placement optimization with multiple global and local methods.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 145.0  # safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta=100.0):
    """Smooth approximation of minimum outer side length."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    projections = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                projections.append(proj)
    if not projections:
        return 0.0
    vals = np.asarray(projections, dtype=float)
    m = np.max(vals)
    if m < -1e10:
        return 0.0
    sm = m + np.log(np.sum(np.exp((vals - m) * beta)) / len(vals)) / beta
    return (2.0 / math.sqrt(3.0)) * sm

# --- Objective function for free placement ---
# Parameters: 12*(x,y,angle) + 2*(center_x, center_y) + 1*(outer_angle) = 39 params

def decode_params(params):
    """Decode parameter vector into inner positions and outer params."""
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append((x, y, ang))
    cx = params[3*N]
    cy = params[3*N+1]
    outer_angle = params[3*N+2]
    return inner, (cx, cy), outer_angle

def objective(params, use_smooth=False):
    """Objective: minimize outer side length with penalty for invalidity."""
    inner, center, outer_angle = decode_params(params)
    
    if use_smooth:
        side = compute_min_outer_side_smooth(inner, center, outer_angle, beta=80.0)
    else:
        side = compute_min_outer_side(inner, center, outer_angle)
    
    # Check validity
    if not check_validity(inner, center, side, outer_angle):
        return side + 50.0  # heavy penalty for invalid
    
    return side

def objective_smooth(params):
    """Smooth objective for gradient-based methods."""
    return objective(params, use_smooth=True)

def objective_exact(params):
    """Exact objective."""
    return objective(params, use_smooth=False)

# --- Initial configuration builders ---

def build_hex_lattice_ring():
    """12 hexagons in a ring pattern on hexagonal lattice."""
    inner = []
    # Ring of 12 at distance ~2.0 from center
    for k in range(12):
        theta = k * math.pi / 6.0
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), 0.0

def build_two_rings():
    """Two concentric rings of 6."""
    inner = []
    for k in range(6):
        theta = k * math.pi / 3.0
        x = 1.8 * math.cos(theta)
        y = 1.8 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(6):
        theta = (k + 0.5) * math.pi / 3.0
        x = 3.2 * math.cos(theta)
        y = 3.2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), 0.0

def build_center_plus_ring():
    """Center hexagon + ring of 12."""
    inner = [(0.0, 0.0, 0.0)]
    for k in range(12):
        theta = k * math.pi / 6.0
        x = 2.2 * math.cos(theta)
        y = 2.2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), 0.0

def build_rect_grid():
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * 2.0
            y = (j - 1.5) * 1.8
            inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), 0.0

def encode_config(inner, center, outer_angle):
    """Encode a configuration into parameter vector."""
    params = []
    for (x, y, ang) in inner:
        params.extend([x, y, ang])
    params.extend([center[0], center[1], outer_angle])
    return np.array(params, dtype=float)

# --- Optimization ---

def run_differential_evolution(bounds, x0, time_budget):
    """Run differential evolution for global optimization."""
    start = time.time()
    
    def de_objective(params):
        if time.time() > DEADLINE - 5.0:
            return 10000.0
        return objective_exact(params)
    
    try:
        result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=100, tol=1e-6,
            popsize=15, mutation=(0.3, 0.9), recombination=0.8,
            polish=True, init='latinhypercube'
        )
        return result.x, result.fun
    except Exception:
        return x0, objective_exact(x0)

def run_local_optimization(x0, bounds, time_budget):
    """Run local optimization from a starting point."""
    start = time.time()
    best_x = x0.copy()
    best_f = objective_exact(x0)
    
    # Nelder-Mead with smooth objective first
    try:
        result = minimize(
            objective_smooth, best_x,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-7}
        )
        if result.fun < best_f:
            best_x = result.x
            best_f = objective_exact(best_x)
    except Exception:
        pass
    
    # Polish with exact objective
    if time.time() < DEADLINE - 10.0:
        try:
            result = minimize(
                objective_exact, best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-9}
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f

def run_coarse_grid_search(bounds, n_per_dim=5, time_budget=30.0):
    """Coarse grid search to find good starting points."""
    start = time.time()
    n_dims = len(bounds)
    
    # Adaptive grid: fewer points per dimension if many dimensions
    points_per_dim = max(3, min(n_per_dim, int(time_budget / (n_dims * 2.0))))
    
    best_x = None
    best_f = float('inf')
    
    # Use random sampling instead of full grid for high dimensions
    n_samples = min(200, max(50, int(time_budget / 0.1)))
    rng = np.random.RandomState(123)
    
    for _ in range(n_samples):
        if time.time() > start + time_budget:
            break
        x = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        f = objective_exact(x)
        if f < best_f:
            best_f = f
            best_x = x
    
    return best_x, best_f

def optimize():
    """Main optimization routine."""
    t_start = time.time()
    
    # Define bounds for all 39 parameters
    # 12 hexagons: x,y in [-6,6], angle in [0,180]
    # Outer: cx,cy in [-4,4], angle in [0,60]
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0))  # x
        bounds.append((-6.0, 6.0))  # y
        bounds.append((0.0, 180.0)) # angle
    bounds.append((-4.0, 4.0))  # outer cx
    bounds.append((-4.0, 4.0))  # outer cy
    bounds.append((0.0, 60.0))  # outer angle
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: hex lattice ring
    inner1, center1, orot1 = build_hex_lattice_ring()
    initial_configs.append(encode_config(inner1, center1, orot1))
    
    # Config 2: two rings
    inner2, center2, orot2 = build_two_rings()
    initial_configs.append(encode_config(inner2, center2, orot2))
    
    # Config 3: center + ring
    inner3, center3, orot3 = build_center_plus_ring()
    initial_configs.append(encode_config(inner3, center3, orot3))
    
    # Config 4: rectangular grid
    inner4, center4, orot4 = build_rect_grid()
    initial_configs.append(encode_config(inner4, center4, orot4))
    
    # Config 5: random
    rng = np.random.RandomState(42)
    for i in range(N):
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        ang = rng.uniform(0, 180)
        inner4[i] = (x, y, ang)
    initial_configs.append(encode_config(inner4, (0,0), 0.0))
    
    # Evaluate initial configs
    best_x = None
    best_f = float('inf')
    for cfg in initial_configs:
        f = objective_exact(cfg)
        if f < best_f:
            best_f = f
            best_x = cfg.copy()
    
    # Phase 1: Differential evolution (global search)
    if time.time() < DEADLINE - 30.0:
        de_budget = min(40.0, DEADLINE - 30.0 - time.time())
        de_x, de_f = run_differential_evolution(bounds, best_x, de_budget)
        if de_f < best_f:
            best_f = de_f
            best_x = de_x
    
    # Phase 2: Local optimization from best point
    if time.time() < DEADLINE - 15.0:
        local_budget = DEADLINE - 15.0 - time.time()
        local_x, local_f = run_local_optimization(best_x, bounds, local_budget)
        if local_f < best_f:
            best_f = local_f
            best_x = local_x
    
    # Phase 3: Multi-start local optimization from different initial configs
    for cfg in initial_configs[:3]:
        if time.time() > DEADLINE - 10.0:
            break
        try:
            result = minimize(
                objective_smooth, cfg,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-7}
            )
            f = objective_exact(result.x)
            if f < best_f:
                best_f = f
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time.time() < DEADLINE - 5.0:
        try:
            result = minimize(
                objective_exact, best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-10}
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Decode best solution
    inner, center, outer_angle = decode_params(best_x)
    
    # Compute exact side length
    side = compute_min_outer_side(inner, center, outer_angle)
    
    # Final validity check
    if not check_validity(inner, center, side, outer_angle):
        # Try to fix by scaling inner positions toward center
        for scale_factor in [0.95, 0.9, 0.85, 0.8]:
            inner_scaled = [(x * scale_factor, y * scale_factor, ang) for (x, y, ang) in inner]
            side_scaled = compute_min_outer_side(inner_scaled, center, outer_angle)
            if check_validity(inner_scaled, center, side_scaled, outer_angle):
                inner = inner_scaled
                side = side_scaled
                break
    
    return inner, center, side, outer_angle

def construct():
    """Entry point: construct the packing."""
    t0 = time.time()
    try:
        inner, center, side, outer_angle = optimize()
        # Final safety check
        if check_validity(inner, center, side, outer_angle):
            return inner, center, side, outer_angle
    except Exception:
        pass
    
    # Fallback: simple ring arrangement
    inner = []
    for k in range(12):
        theta = k * math.pi / 6.0
        x = 2.5 * math.cos(theta)
        y = 2.5 * math.sin(theta)
        inner.append((x, y, 0.0))
    side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), side, 0.0
