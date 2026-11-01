"""Packing 12 unit hexagons in minimal outer hexagon using simulated annealing."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def convex_hull_area(pts):
    """Compute area of convex hull of points."""
    if len(pts) < 3:
        return 0.0
    try:
        h = ConvexHull(pts)
        return h.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def max_dist_from_center(pts, center):
    """Max distance from center to any point."""
    if len(pts) == 0:
        return 0.0
    return np.max(np.linalg.norm(pts - center, axis=1))

def compute_outer_hexagon(inner_hexes):
    """
    Given list of (cx, cy, angle) for inner hexagons, find minimal outer hexagon.
    We try to fit all hex vertices in a regular hexagon, minimizing its side length.
    Use the fact that for a regular hexagon, the minimal containing regular hexagon
    can be found by trying different orientations.
    """
    # Collect all vertices of all inner hexagons
    all_pts = []
    for (cx, cy, ang) in inner_hexes:
        v = hex_vertices(cx, cy, 1.0, ang)
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    
    # Center of mass as approximate center
    center = np.mean(all_pts, axis=0)
    
    # Try different orientations of the outer hexagon
    best_side = float('inf')
    best_center = center
    best_angle = 0.0
    
    for outer_angle_deg in range(0, 180, 5):
        outer_a = np.radians(outer_angle_deg)
        # For a regular hexagon with side s centered at c with rotation a,
        # a point p is inside iff for all 6 face normals, dot(p-c, n_i) <= s * cos(pi/6)
        # Actually, the distance from center to each edge is s * cos(pi/6) = s * sqrt(3)/2
        # The 6 outward normals of a regular hexagon (pointing outward from center)
        # are at angles outer_a + k*pi/3 for k=0..5, but these are vertex directions.
        # Edge normals are at outer_a + pi/6 + k*pi/3
        
        # Let me reconsider: A regular hexagon with vertices at angles theta_k = outer_a + k*pi/3
        # has edges whose outward normals are at angles outer_a + pi/6 + k*pi/3
        # The distance from center to each edge (apothem) is s * cos(pi/6) = s * sqrt(3)/2
        
        apothem_needed = 0.0
        for k in range(6):
            normal_angle = outer_a + np.pi/6 + k * np.pi/3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # Max projection of (p - center) onto this normal
            projections = (all_pts[:, 0] - center[0]) * nx + (all_pts[:, 1] - center[1]) * ny
            max_proj = np.max(projections)
            # Need apothem >= max_proj, so s >= max_proj / cos(pi/6) = max_proj * 2/sqrt(3)
            if max_proj > 0:
                s_needed = max_proj * 2.0 / np.sqrt(3.0)
                if s_needed > apothem_needed:
                    apothem_needed = s_needed
        
        s = apothem_needed
        if s < best_side:
            best_side = s
            best_center = center
            best_angle = outer_angle_deg
    
    return best_center, best_side, best_angle

def evaluate_configuration(inner_hexes):
    """Return the outer hexagon side length for a configuration."""
    center, side, angle = compute_outer_hexagon(inner_hexes)
    return side, center, angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    np.random.seed(42)
    
    # Initial configuration: place hexagons in a rough grid
    # Try a hexagonal lattice arrangement
    def initial_config():
        configs = []
        # Try several initial arrangements
        # 1) Linear
        cfg1 = []
        for i in range(N):
            cfg1.append([i * 1.8, 0.0, 0.0])
        configs.append(cfg1)
        
        # 2) Hexagonal cluster (center + 2 rings)
        cfg2 = []
        # Center
        cfg2.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons at distance ~1.8
        for k in range(6):
            ang = k * np.pi / 3.0
            cfg2.append([1.8 * np.cos(ang), 1.8 * np.sin(ang), 0.0])
        # Ring 2: 5 more hexagons (we need 12 total, have 7, need 5 more)
        for k in range(5):
            ang = k * 2 * np.pi / 5.0 + 0.3
            cfg2.append([3.2 * np.cos(ang), 3.2 * np.sin(ang), 0.0])
        configs.append(cfg2)
        
        # 3) Random
        cfg3 = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            ang = np.random.uniform(0, 60)
            cfg3.append([x, y, ang])
        configs.append(cfg3)
        
        return configs
    
    all_configs = initial_config()
    
    # Evaluate all initial configs
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_config = None
    
    for cfg in all_configs:
        side, center, angle = evaluate_configuration(cfg)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = angle
            best_config = [list(c) for c in cfg]
    
    # Simulated annealing
    current_config = [list(c) for c in best_config]
    current_side = best_side
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    max_steps = 50000
    
    step = 0
    while step < max_steps and time.time() < deadline:
        # Perturb one hexagon
        idx = np.random.randint(N)
        dx = np.random.normal(0, 0.3)
        dy = np.random.normal(0, 0.3)
        dangle = np.random.normal(0, 5.0)
        
        old_x, old_y, old_ang = current_config[idx]
        new_x = old_x + dx
        new_y = old_y + dy
        new_ang = (old_ang + dangle) % 60.0  # Hexagon symmetry: 60 degrees
        
        current_config[idx] = [new_x, new_y, new_ang]
        
        new_side, new_center, new_angle = evaluate_configuration(current_config)
        
        if new_side < current_side or np.random.random() < np.exp((current_side - new_side) / T):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_center = new_center
                best_angle = new_angle
                best_config = [list(c) for c in current_config]
        else:
            # Revert
            current_config[idx] = [old_x, old_y, old_ang]
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        step += 1
        
        # Check time more frequently
        if step % 1000 == 0 and time.time() > deadline - 5:
            break
    
    return best_config, best_center, best_side, best_angle
