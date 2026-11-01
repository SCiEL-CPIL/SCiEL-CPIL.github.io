"""
Physics-based relaxation using Lennard-Jones-like repulsion between hexagons.
The idea: model each hexagon as a circle of effective radius, use smooth
repulsive potentials to avoid overlaps, and minimize the outer hexagon side.
"""
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
            s = sat_min_separation(centers[i], centers[j])
            if s < -1e-9:
                return False
    return True

def physics_relaxation(init_centers, time_limit=60.0):
    """
    Physics-based relaxation using smooth repulsive potentials.
    Each hexagon is treated as a disk of effective radius ~1.0 (circumradius).
    We use a Lennard-Jones-like potential for repulsion and minimize
    the outer hexagon side length.
    """
    start_time = time.time()
    
    # Effective radius for repulsion: use circumradius = 1.0 for unit hexagons
    # But hexagons are not circles; use a slightly larger effective radius
    # to account for the flat sides. The minimum distance between centers
    # for non-overlapping hexagons depends on orientation.
    # Use r_eff = 1.0 as a starting point; the potential will push them apart.
    r_eff = 1.0
    
    # Parameters: x, y, angle for each hexagon
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = init_centers[i][0]
        params[3*i+1] = init_centers[i][1]
        params[3*i+2] = init_centers[i][2]
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e6
        
        # Extract centers
        xs = p[0::3]
        ys = p[1::3]
        angles = p[2::3]
        
        # Compute repulsive potential between all pairs
        # Use a smooth approximation: for distance d, if d < 2*r_eff, repel
        # Potential: sum over pairs of max(0, 2*r_eff - d)^2 * strength
        repulsion = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                # For hexagons, the minimum center distance depends on relative angle
                # Use a conservative estimate: 2*r_eff = 2.0
                # But we can be smarter: the actual minimum distance for two unit
                # hexagons at relative angle delta is between 2*apothem and 2*circumradius
                # apothem = sqrt(3)/2 ~ 0.866, circumradius = 1.0
                # So min distance is between 1.732 and 2.0
                # Use 1.8 as a compromise
                min_dist = 1.8
                if d < min_dist:
                    repulsion += (min_dist - d) ** 2
        
        # Outer side length
        centers = []
        for i in range(N):
            centers.append([xs[i], ys[i], angles[i]])
        req_side = required_outer_side(centers, (0.0, 0.0), 0.0)
        
        # Total objective: outer side + penalty for overlaps
        # Scale repulsion so it's significant
        return req_side + 10.0 * repulsion
    
    # Use L-BFGS-B with bounds
    # Bounds: positions in [-10, 10], angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 60.0))    # angle
    
    result = minimize(
        objective, params, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    # For unit hexagons, the minimum center distance is ~1.732 (2*apothem)
    # Hexagonal lattice spacing: a = 2*apothem = sqrt(3) ~ 1.732
    a = np.sqrt(3)
    
    # 12 hexagons in a hexagonal arrangement
    # Ring 0: 1 center
    # Ring 1: 6 hexagons
    # Ring 2: 5 more hexagons to make 12
    positions = [
        (0, 0),
        (a, 0), (-a, 0),
        (a/2, a*np.sqrt(3)/2), (-a/2, a*np.sqrt(3)/2),
        (a/2, -a*np.sqrt(3)/2), (-a/2, -a*np.sqrt(3)/2),
        (3*a/2, a*np.sqrt(3)/2), (-3*a/2, a*np.sqrt(3)/2),
        (3*a/2, -a*np.sqrt(3)/2), (-3*a/2, -a*np.sqrt(3)/2),
        (0, a*np.sqrt(3)), (0, -a*np.sqrt(3)),
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = (np.degrees(theta)) % 60
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

def initial_config_tight_hex():
    """Create a tight hexagonal packing with minimal spacing."""
    centers = []
    # Use the minimum possible spacing for hexagonal packing
    # Center distance = 2 * apothem = sqrt(3) ~ 1.732
    a = np.sqrt(3)
    
    # Hexagonal numbers: 1, 7, 19...
    # Use 1 center + 6 in first ring + 5 in second ring
    positions = [
        (0, 0),
        (a, 0), (-a, 0),
        (a*0.5, a*0.866), (-a*0.5, a*0.866),
        (a*0.5, -a*0.866), (-a*0.5, -a*0.866),
        (a*1.5, a*0.866), (-a*1.5, a*0.866),
        (a*1.5, -a*0.866), (-a*1.5, -a*0.866),
        (0, a*1.732), (0, -a*1.732),
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hex", initial_config_hexagonal()),
        ("tight_hex", initial_config_tight_hex()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(777)),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        try:
            # Physics-based relaxation
            relax_time = min(25.0, DEADLINE - elapsed - 5)
            if relax_time < 5:
                relax_time = 5
            relaxed = physics_relaxation(config, time_limit=relax_time)
        except Exception:
            relaxed = [list(c) for c in config]
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Compute final required side
        final_side = required_outer_side(relaxed, (0.0, 0.0), 0.0)
        
        # Validate: check no overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(relaxed[i], relaxed[j])
                if s < -1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and final_side < best_score:
            best_score = final_side
            best_result = (relaxed, (0.0, 0.0), final_side, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback: hexagonal lattice
    centers = initial_config_hexagonal()
    required_side = required_outer_side(centers, (0.0, 0.0), 0.0)
    return centers, (0.0, 0.0), required_side, 0.0
