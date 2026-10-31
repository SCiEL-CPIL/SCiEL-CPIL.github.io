"""Circle packing in unit square: 26 circles, maximize sum of radii.
Combined approach: Voronoi/Delaunoi-based structural initialization + force-directed
refinement + vectorized objective + DE polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay, Voronoi

N = 26
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _clamp_circles(c):
    """Ensure circles are inside [0,1]^2 by shrinking radii and clamping centers."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    r = np.maximum(r, 1e-6)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return np.column_stack([x, y, r])

def _penalty(c, w_overlap=1000.0, w_boundary=1000.0, w_small=1.0):
    """Penalty for invalid packing. Lower is better (0 = valid)."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    n = len(c)
    pen = 0.0
    # Boundary violations
    for i in range(n):
        if r[i] <= 0:
            pen += w_small * (1e-6 - r[i]) ** 2
        for bound in (x[i], y[i], 1 - x[i], 1 - y[i]):
            if bound < r[i]:
                pen += w_boundary * (r[i] - bound) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            need = r[i] + r[j]
            if d < need:
                pen += w_overlap * (need - d) ** 2
    return pen

def _objective(c):
    """Maximize sum of radii => minimize -(sum r) + penalty."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    r = c[:, 2]
    return -np.sum(r) + _penalty(c)

def _is_valid(c, tol=1e-6):
    c = np.array(c, dtype=float).reshape(-1, 3)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    n = len(c)
    if np.any(r <= 0):
        return False
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol or y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - tol:
                return False
    return True

def _voronoi_init(rng):
    """
    Generate initial packing using Voronoi tessellation.
    Place seed points, compute Voronoi cells, set radius to min distance to cell boundary.
    Then refine seeds via gradient ascent on sum of Voronoi-derived radii.
    """
    # Start with a jittered grid
    side = 5
    seeds = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5 + rng.uniform(-0.15, 0.15)) / side
            y = (j + 0.5 + rng.uniform(-0.15, 0.15)) / side
            seeds.append([x, y])
    # Add one more seed
    seeds.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
    seeds = np.array(seeds[:N])
    
    # Add boundary points to help define the Voronoi cells within the unit square
    # We add points well outside the unit square so that the Voronoi cells
    # for our seeds are bounded by the unit square edges
    boundary_points = []
    for i in range(20):
        angle = 2 * np.pi * i / 20
        boundary_points.append([0.5 + 2.0 * np.cos(angle), 0.5 + 2.0 * np.sin(angle)])
    boundary_points = np.array(boundary_points)
    
    def _voronoi_radii(seeds):
        """Compute radii from Voronoi cells: min distance from seed to cell boundary."""
        all_points = np.vstack([seeds, boundary_points])
        try:
            vor = Voronoi(all_points)
        except Exception:
            return np.full(N, 0.05)
        
        radii = np.zeros(N)
        for i in range(N):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            if len(region) == 0 or -1 in region:
                # Unbounded cell (shouldn't happen with boundary points)
                radii[i] = 0.05
                continue
            
            # Get the vertices of the Voronoi cell
            vertices = vor.vertices[region]
            if len(vertices) == 0:
                radii[i] = 0.05
                continue
            
            # Compute min distance from seed to each edge of the cell
            min_dist = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                # Distance from point to line segment
                px, py = seeds[i]
                x1, y1 = v1
                x2, y2 = v2
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    d = np.sqrt((px - x1)**2 + (py - y1)**2)
                else:
                    t = np.clip((dx * (px - x1) + dy * (py - y1)) / (dx**2 + dy**2), 0, 1)
                    cx, cy = x1 + t * dx, y1 + t * dy
                    d = np.sqrt((px - cx)**2 + (py - cy)**2)
                min_dist = min(min_dist, d)
            
            # Also check distance to unit square boundaries
            min_dist = min(min_dist, seeds[i][0], 1 - seeds[i][0], seeds[i][1], 1 - seeds[i][1])
            radii[i] = max(min_dist, 0.01)
        
        return radii
    
    # Initial radii
    radii = _voronoi_radii(seeds)
    
    # Refine seeds via gradient ascent on sum of radii
    # Use finite differences for gradient
    def _sum_radii(seeds):
        return np.sum(_voronoi_radii(seeds))
    
    best_seeds = seeds.copy()
    best_val = _sum_radii(best_seeds)
    
    lr = 0.01
    for iteration in range(50):
        if _remaining() < 15:
            break
        # Compute gradient via finite differences
        grad = np.zeros_like(seeds)
        eps = 0.001
        for i in range(N):
            for d in range(2):
                s_plus = seeds.copy()
                s_minus = seeds.copy()
                s_plus[i, d] += eps
                s_minus[i, d] -= eps
                # Keep within bounds
                s_plus[i, d] = np.clip(s_plus[i, d], 0.01, 0.99)
                s_minus[i, d] = np.clip(s_minus[i, d], 0.01, 0.99)
                f_plus = _sum_radii(s_plus)
                f_minus = _sum_radii(s_minus)
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Gradient ascent step
        seeds = seeds + lr * grad
        seeds = np.clip(seeds, 0.01, 0.99)
        
        val = _sum_radii(seeds)
        if val > best_val:
            best_val = val
            best_seeds = seeds.copy()
            lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.7
    
    radii = _voronoi_radii(best_seeds)
    return np.column_stack([best_seeds, radii])

def _force_directed_init(rng):
    """
    Generate initial packing using force-directed relaxation.
    Start with random positions, apply repulsive forces between circles
    and attractive forces toward center, iterate until stable.
    """
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = np.full(N, 0.08)
    
    for iteration in range(200):
        if _remaining() < 15:
            break
        forces = np.zeros((N, 2))
        
        for i in range(N):
            # Repulsive forces from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-8:
                    # Strong repulsion
                    f = 10.0 * (min_dist - dist) / dist
                    forces[i][0] += f * dx
                    forces[i][1] += f * dy
                elif dist < 2 * min_dist and dist > 1e-8:
                    # Mild repulsion to spread out
                    f = 0.5 * (2 * min_dist - dist) / dist
                    forces[i][0] += f * dx
                    forces[i][1] += f * dy
            
            # Attractive force toward center (weak)
            forces[i][0] += 0.1 * (0.5 - pos[i][0])
            forces[i][1] += 0.1 * (0.5 - pos[i][1])
        
        # Update positions
        pos = pos + 0.1 * forces
        pos = np.clip(pos, 0.05, 0.95)
        
        # Update radii based on min distance to neighbors and walls
        for i in range(N):
            min_dist = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Max radius such that circles don't overlap
                max_r = (dist - radii[j]) / 2
                min_dist = min(min_dist, max(0.01, max_r))
            radii[i] = min(radii[i] * 1.05, min_dist)
    
    return np.column_stack([pos, radii])

def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    return np.array(cells[:N])

def _hex_init():
    """Hexagonal-ish lattice init."""
    cells = []
    rows = int(np.ceil(np.sqrt(N)))
    for j in range(rows):
        for i in range(rows):
            x = (i + 0.5 + (0.25 if j % 2 else 0)) / (rows + 0.25)
            y = (j + 0.5) / (rows + 0.25)
            cells.append([x, y, 0.5 / (rows + 0.25) - 1e-4])
    return np.array(cells[:N])

def _random_init(rng):
    cells = []
    for _ in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.02, 0.12)
        cells.append([x, y, r])
    return np.array(cells)

def _local_refine(c, max_iter=200, tol=1e-8):
    """Local L-BFGS-B refinement with penalty."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    bounds = []
    for i in range(len(c)):
        bounds.append((1e-6, 1 - 1e-6))  # x
        bounds.append((1e-6, 1 - 1e-6))  # y
        bounds.append((1e-6, 0.5))       # r
    try:
        res = minimize(_objective, c.ravel(), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if res.success or res.fun < _objective(c.ravel()):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return c

def _polish_de(c, rng, time_budget=15.0):
    """Local differential evolution polishing around current best."""
    c = np.array(c, dtype=float).reshape(-1, 3)
    n = len(c)
    bounds = []
    for i in range(n):
        cx, cy, cr = c[i]
        bounds.append((max(1e-6, cx - 0.15), min(1 - 1e-6, cx + 0.15)))
        bounds.append((max(1e-6, cy - 0.15), min(1 - 1e-6, cy + 0.15)))
        bounds.append((max(1e-6, cr * 0.5), min(0.5, cr * 1.8)))
    start = time.time()
    def wrapper(x):
        if time.time() - start > time_budget:
            return 1e6
        return _objective(x)
    try:
        res = differential_evolution(wrapper, bounds, seed=int(rng.integers(0, 1e6)),
                                     maxiter=60, popsize=10, tol=1e-5,
                                     mutation=(0.3, 1.0), recombination=0.8,
                                     init='sohal')
        if res.fun < _objective(c.ravel()):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return c

def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(20240517)

    # Candidate initializations - include new structural methods
    inits = [
        _grid_init(),
        _hex_init(),
    ]
    
    # Add Voronoi-based initialization
    if _remaining() > 30:
        try:
            inits.append(_voronoi_init(rng))
        except Exception:
            pass
    
    # Add force-directed initialization
    if _remaining() > 30:
        try:
            inits.append(_force_directed_init(rng))
        except Exception:
            pass
    
    for _ in range(4):
        inits.append(_random_init(rng))

    # Stage 1: local refine each init
    for c0 in inits:
        if _remaining() < 10:
            break
        c = _clamp_circles(c0)
        c = _local_refine(c, max_iter=300)
        c = _clamp_circles(c)
        val = -_objective(c)  # approx sum r minus penalty
        if _is_valid(c) and np.sum(c[:, 2]) > best_val:
            best_val = np.sum(c[:, 2])
            best = c.copy()

    # Stage 2: DE polishing on best candidates
    if best is not None and _remaining() > 25:
        c = _polish_de(best, rng, time_budget=min(20.0, max(10.0, _remaining() - 15)))
        c = _clamp_circles(c)
        c = _local_refine(c, max_iter=150)
        c = _clamp_circles(c)
        if _is_valid(c) and np.sum(c[:, 2]) > best_val:
            best_val = np.sum(c[:, 2])
            best = c.copy()

    # Stage 3: a few more random restarts if time permits
    if best is not None and _remaining() > 40:
        for _ in range(6):
            if _remaining() < 15:
                break
            c0 = _random_init(rng)
            c = _clamp_circles(c0)
            c = _local_refine(c, max_iter=200)
            c = _clamp_circles(c)
            if _is_valid(c) and np.sum(c[:, 2]) > best_val:
                best_val = np.sum(c[:, 2])
                best = c.copy()
            if _remaining() > 30:
                c = _polish_de(best, rng, time_budget=min(12.0, max(6.0, _remaining() - 15)))
                c = _clamp_circles(c)
                c = _local_refine(c, max_iter=100)
                c = _clamp_circles(c)
                if _is_valid(c) and np.sum(c[:, 2]) > best_val:
                    best_val = np.sum(c[:, 2])
                    best = c.copy()

    if best is None:
        # Fallback: always-valid grid
        best = _grid_init()

    return _clamp_circles(best)
