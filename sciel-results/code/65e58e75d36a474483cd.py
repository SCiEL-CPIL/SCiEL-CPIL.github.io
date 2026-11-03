"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices at angles angle_deg + k*60 for k=0..5, radius = side
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_enclosing_hex_side(points, center=None):
    """
    Given a set of 2D points, find the minimal side length of a regular hexagon
    that encloses them. Returns (min_side, best_rotation_deg, best_center).
    Uses support function sweep over rotation angle.
    """
    points = np.array(points)
    if center is None:
        center = points.mean(axis=0)
    
    # The 6 normal directions for a regular hexagon with rotation theta
    # are at angles theta + k*pi/3
    # For a hexagon with side s, inradius = s * sqrt(3)/2
    # Constraint: max_i dot(p_i - c, u_k) <= s * sqrt(3)/2 for all k
    # So s >= (2/sqrt(3)) * max_k max_i dot(p_i - c, u_k)
    
    sqrt3 = np.sqrt(3.0)
    
    def compute_side(theta):
        """Compute required side length for rotation theta."""
        # 6 outward normal directions
        angles = theta + np.arange(6) * np.pi / 3
        u = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project points onto each direction
        # dots[i, k] = dot(points[i] - center, u[k])
        dots = (points - center) @ u.T  # shape (n_points, 6)
        max_proj = dots.max(axis=0)  # shape (6,)
        max_abs_proj = np.max(np.abs(max_proj))  # we need max of absolute values? 
        # Actually, we need max of dot for each direction (outward), so just max_proj
        # But we should also consider that points could be on the "negative" side
        # The constraint is: for outward normal u_k, all points must satisfy dot(p-c, u_k) <= s*sqrt3/2
        # So s >= (2/sqrt3) * max_k max_i dot(p_i - c, u_k)
        
        required = max_proj.max()
        return (2.0 / sqrt3) * required
    
    # Sweep theta from 0 to pi/3 (60 degrees, since hexagon has 6-fold symmetry)
    # Use fine grid
    thetas = np.linspace(0, np.pi / 3, 3600)
    sides = np.array([compute_side(t) for t in thetas])
    
    min_idx = np.argmin(sides)
    min_side = sides[min_idx]
    best_theta = thetas[min_idx]
    
    # Refine with finer search around best
    theta_refine = np.linspace(best_theta - np.pi/180, best_theta + np.pi/180, 1000)
    sides_refine = np.array([compute_side(t) for t in theta_refine])
    min_idx2 = np.argmin(sides_refine)
    min_side = sides_refine[min_idx2]
    best_theta = theta_refine[min_idx2]
    
    return min_side, np.degrees(best_theta), center

def generate_cluster_configs():
    """
    Generate various cluster configurations of 11 unit hexagons on a hexagonal lattice.
    Returns list of (positions, description) where positions are (x,y) coords of hex centers.
    """
    configs = []
    
    # Hexagonal lattice basis: neighbors at 60 degrees, spacing sqrt(3)
    # For unit hexagons, center-to-center distance for edge-sharing is sqrt(3)
    sqrt3 = np.sqrt(3.0)
    
    # Direction vectors for hex lattice
    dirs = []
    for k in range(6):
        angle = k * np.pi / 3
        dirs.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    
    def lattice_to_xy(q, r):
        """Convert axial coordinates (q,r) to Cartesian for hex lattice with spacing sqrt(3)."""
        x = sqrt3 * (q + r / 2.0)
        y = sqrt3 * (r * np.sqrt(3) / 2.0)  # Hmm, let me reconsider
        # Standard hex grid: 
        # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
        # y = size * (3/2 * r)
        # For unit hexagons, "size" (center to vertex) = 1, so center-to-center = sqrt(3)
        # Actually, let's use: x = sqrt(3)*(q + r/2), y = 1.5*r ... no
        # Let me just use the 6 neighbor directions directly.
        return (x, y)
    
    # Simpler: use the 6 direction vectors to build clusters
    def build_cluster(steps):
        """Build cluster from a list of (q, r) axial coords."""
        # Convert axial to Cartesian
        # x = sqrt(3) * (q + r/2)
        # y = (3/2) * r  ... but this gives wrong spacing
        # Let me verify: neighbor at (1,0) should be at distance sqrt(3)
        # x = sqrt(3)*(1+0) = sqrt(3), y=0. Distance = sqrt(3). Correct.
        # Neighbor at (0,1): x = sqrt(3)/2, y = 3/2. Distance = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
        positions = []
        for (q, r) in steps:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
        return positions
    
    # Configuration 1: 3-4-3-1 arrangement (roughly hexagonal)
    # Center hex at (0,0), then surrounding
    # A hex number: 1, 7, 19, 37... We need 11.
    # Start with center + 6 neighbors = 7, then add 4 more.
    
    # Config A: Center + 6 neighbors + 4 additional in a row
    config_a = [(0,0)]
    for k in range(6):
        # 6 neighbors in axial coords
        neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        config_a.append(neighbors[k])
    # Add 4 more: extend in one direction
    # From (1,0), add (2,0), (3,0), (4,0), (5,0) - but that's a long tail
    # Better: add (2,0), (2,-1), (1,-2), (0,-2) to make a thicker shape
    config_a += [(2,0), (2,-1), (1,-2), (0,-2)]
    # Check for duplicates
    if len(set(config_a)) != 11:
        # Remove duplicates
        seen = set()
        unique = []
        for p in config_a:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        config_a = unique[:11]
    configs.append((build_cluster(config_a), "center+6+4_tail"))
    
    # Config B: More compact - 2 rows of 5 and 1 row of 1, or similar
    # Try: row at y=0: 5 hexes, row at y=1.5: 5 hexes, row at y=3: 1 hex
    # In axial: 
    # Row r=0: q = -2,-1,0,1,2 -> 5 hexes
    # Row r=1: q = -2,-1,0,1,2 -> 5 hexes (shifted)
    # Row r=2: q = 0 -> 1 hex
    config_b = []
    for q in range(-2, 3):
        config_b.append((q, 0))
    for q in range(-2, 3):
        config_b.append((q, 1))
    config_b.append((0, 2))
    configs.append((build_cluster(config_b), "5+5+1_rows"))
    
    # Config C: 3 rows: 4+4+3
    config_c = []
    for q in range(-1, 3):  # 4 hexes
        config_c.append((q, 0))
    for q in range(-1, 3):  # 4 hexes
        config_c.append((q, 1))
    for q in range(-1, 2):  # 3 hexes
        config_c.append((q, 2))
    configs.append((build_cluster(config_c), "4+4+3_rows"))
    
    # Config D: Compact blob - center + 6 + 4 in two opposite directions
    config_d = [(0,0)]
    neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    config_d += neighbors
    # Add (2,0), (2,-1), (-2,0), (-2,1)
    config_d += [(2,0), (2,-1), (-2,0), (-2,1)]
    if len(set(config_d)) == 11:
        configs.append((build_cluster(config_d), "center+6+2opposite"))
    
    # Config E: 2x2x2 block plus extras - not quite, let's do a diamond
    # Diamond shape: 1+2+3+2+1 = 9, need 2 more
    config_e = []
    # Row r=-2: 1 hex at q=0
    config_e.append((0, -2))
    # Row r=-1: 2 hexes at q=-1,0
    config_e.append((-1, -1))
    config_e.append((0, -1))
    # Row r=0: 3 hexes at q=-1,0,1
    config_e.append((-1, 0))
    config_e.append((0, 0))
    config_e.append((1, 0))
    # Row r=1: 2 hexes at q=0,1
    config_e.append((0, 1))
    config_e.append((1, 1))
    # Row r=2: 1 hex at q=1
    config_e.append((1, 2))
    # That's 9. Add 2 more: (2,0) and (2,1)
    config_e.append((2, 0))
    config_e.append((2, 1))
    if len(set(config_e)) == 11:
        configs.append((build_cluster(config_e), "diamond+2"))
    
    # Config F: Long hexagonal strip 1+2+2+2+2+1 = 10, add 1
    config_f = []
    config_f.append((0, 0))
    for r in range(1, 5):
        config_f.append((0, r))
        config_f.append((1, r))
    config_f.append((1, 5))
    # That's 1+4*2+1 = 10. Add (2,2)
    config_f.append((2, 2))
    if len(set(config_f)) == 11:
        configs.append((build_cluster(config_f), "strip+1"))
    
    return configs

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    configs = generate_cluster_configs()
    
    best_side = float('inf')
    best_result = None
    
    for positions, desc in configs:
        if time.time() > deadline - 5.0:
            break
        
        points = np.array(positions)
        
        # Compute minimal enclosing hexagon
        min_side, best_theta, center = min_enclosing_hex_side(points, center=None)
        
        if min_side < best_side:
            best_side = min_side
            # Build the inner hexagon list
            # Each inner hexagon is at position with angle 0 (they're on a hex lattice, so aligned)
            # But we can rotate the whole cluster... actually the inner hexagons can have any rotation.
            # For a hex lattice packing, all inner hexagons are aligned (angle 0) is optimal.
            # The outer hexagon has rotation best_theta.
            
            # Center the cluster at origin for the inner positions, then the outer center is (0,0)
            # Actually, let's keep the computed center
            centroid = points.mean(axis=0)
            # Shift points so centroid is at origin
            shifted_points = points - centroid
            
            inner = []
            for (x, y) in shifted_points:
                inner.append([x, y, 0.0])
            
            # Outer hexagon center is at (0,0) since we shifted to centroid
            # Outer side length is min_side, rotation is best_theta
            best_result = (inner, (0.0, 0.0), min_side, np.degrees(best_theta))
    
    if best_result is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

