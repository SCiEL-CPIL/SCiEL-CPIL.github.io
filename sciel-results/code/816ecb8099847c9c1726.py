"""Packing 11 unit hexagons in minimal outer hexagon via lattice cluster search."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of unit hexagon centered at (cx,cy) with given rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + math.radians(60 * k)
        verts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return verts

def lattice_position(i, j):
    """Position of hexagon at axial lattice coords (i,j) for unit hexagons (side=1)."""
    x = math.sqrt(3) * (i + j / 2.0)
    y = 1.5 * j
    return (x, y)

def enclosing_hexagon_side(points, center, angle_deg):
    """
    Minimum side length of regular hexagon with given center and rotation
    that contains all points.
    """
    angle = math.radians(angle_deg)
    apothem_factor = math.sqrt(3) / 2.0
    max_proj = 0.0
    for k in range(6):
        theta = angle + math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        for (px, py) in points:
            dx = px - center[0]
            dy = py - center[1]
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
    if max_proj <= 0:
        return 0.0
    return max_proj / apothem_factor

def optimize_enclosing_hex(points):
    """Find minimum enclosing regular hexagon for given set of points."""
    pts = np.array(points)
    cx0 = float(np.mean(pts[:, 0]))
    cy0 = float(np.mean(pts[:, 1]))
    
    def objective(params):
        cx, cy, angle_deg = params
        return enclosing_hexagon_side(points, (cx, cy), angle_deg)
    
    x0 = np.array([cx0, cy0, 0.0])
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
    return float(result.fun), result.x

def get_neighbors(i, j):
    return [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j-1), (i-1, j+1)]

def generate_clusters():
    """Generate candidate 11-hexagon connected clusters on hexagonal lattice."""
    clusters = []
    
    def greedy_growth(seed):
        cluster = [seed]
        used = {seed}
        for _ in range(N - 1):
            candidates = []
            for site in cluster:
                for nb in get_neighbors(*site):
                    if nb not in used:
                        candidates.append(nb)
            if not candidates:
                break
            best = None
            best_score = float('inf')
            for cand in candidates:
                new_cluster = cluster + [cand]
                pos = [lattice_position(i, j) for (i, j) in new_cluster]
                cx = np.mean([p[0] for p in pos])
                cy = np.mean([p[1] for p in pos])
                max_d = max(math.hypot(p[0]-cx, p[1]-cy) for p in pos)
                if max_d < best_score:
                    best_score = max_d
                    best = cand
            if best is None:
                break
            cluster.append(best)
            used.add(best)
        return cluster if len(cluster) == N else None
    
    seeds = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2),
             (-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2)]
    
    for seed in seeds:
        cl = greedy_growth(seed)
        if cl is not None:
            clusters.append(cl)
    
    # Hand-crafted: center + ring1(6) + 4 from ring2
    center = (0, 0)
    ring1 = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    ring2_samples = [
        [(2,0), (1,1), (0,2), (-1,1)],
        [(2,0), (1,1), (2,-1), (1,-2)],
        [(1,0), (2,-1), (1,-2), (0,-2)],
        [(-1,0), (-2,1), (-2,0), (-1,-1)],
        [(0,1), (1,1), (0,2), (-1,2)],
        [(-1,-1), (-2,-1), (-2,0), (-3,0)],
        [(1,1), (0,2), (-1,2), (-2,1)],
        [(2,-1), (1,-2), (0,-2), (-1,-2)],
    ]
    for extra in ring2_samples:
        cl = [center] + ring1 + extra
        if len(cl) == N and len(set(cl)) == N:
            clusters.append(cl)
    
    # Deduplicate
    unique = []
    seen = set()
    for cl in clusters:
        key = tuple(sorted(cl))
        if key not in seen:
            seen.add(key)
            unique.append(cl)
    
    return unique

def construct():
    clusters = generate_clusters()
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for cluster in clusters:
        centers = [lattice_position(i, j) for (i, j) in cluster]
        
        all_points = []
        for (cx, cy) in centers:
            verts = hex_vertices(cx, cy, 0.0)
            all_points.extend(verts)
        
        side, params = optimize_enclosing_hex(all_points)
        
        if side < best_side:
            best_side = side
            best_center = (float(params[0]), float(params[1]))
            best_angle = float(params[2])
            best_inner = [[float(cx), float(cy), 0.0] for (cx, cy) in centers]
    
    if best_inner is None:
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
