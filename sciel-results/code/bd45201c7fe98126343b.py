# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_lattice_positions():
    """
    Return 12 positions of unit hexagons on a hexagonal lattice.
    Hexagons tile with centers at:
      (i * 2.5 + j * 1.25, j * sqrt(3) * 1.25) for appropriate (i,j)
    We pick 12 positions that form a compact cluster.
    """
    sqrt3 = np.sqrt(3)
    # Hexagonal lattice: centers at (2.5*i + 1.25*j, sqrt(3)*1.25*j)
    # Pick 12 positions forming a hexagonal cluster of radius 2
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i)
        x = 2.5 * np.cos(angle)
        y = 2.5 * np.sin(angle)
        positions.append((x, y))
    # Ring 2: 5 more hexagons (to get 12 total)
    # Select 5 from the 12 positions in ring 2
    for i in range(6):
        angle = np.radians(60 * i + 30)
        x = 5.0 * np.cos(angle)
        y = 5.0 * np.sin(angle)
        positions.append((x, y))
    # Take first 12
    return np.array(positions[:12])

def _get_all_inner_vertices(positions):
    """Return (72, 2) array of all vertices of 12 unit hexagons."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        cx, cy = positions[i]
        all_verts[i*6:(i+1)*6] = _hex_vertices(cx, cy, 1.0, 0)
    return all_verts

def _min_containing_hex_side(positions, ox=0.0, oy=0.0, angle_deg=0.0):
    """
    Compute minimum side s of outer hexagon centered at (ox,oy) with given angle
    that contains all inner hexagons.
    """
    all_verts = _get_all_inner_vertices(positions)
    a_rad = np.radians(angle_deg)
    
    def _is_inside(s):
        outer_verts = _hex_vertices(ox, oy, s, angle_deg)
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            edge = p2 - p1
            diffs = all_verts - p1
            crosses = edge[0] * diffs[:, 1] - edge[1] * diffs[:, 0]
            if np.any(crosses < -1e-6):
                return False
        return True
    
    lo = np.sqrt(12)
    hi = 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _is_inside(mid):
            hi = mid
        else:
            lo = mid
    return hi

def _check_separation_fast(positions):
    """Quick check: minimum distance between any pair of centers >= 2.0 (for unit hexagons)."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0 - 1e-6:
                return False
    return True

def _objective(params):
    """
    params: [ox, oy, angle_deg] - outer hexagon center and angle
    Returns the minimal side length s.
    """
    ox, oy, angle_deg = params
    positions = _hex_lattice_positions()
    # Shift positions so their centroid is at origin for symmetry
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    
    s = _min_containing_hex_side(positions, ox, oy, angle_deg)
    return s

def hexagon_packing_12():
    """
    Pack 12 unit hexagons (on a hexagonal lattice, non-overlapping by construction)
    inside a minimal regular hexagon. Optimize outer hexagon center and angle.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Get lattice positions and center them
    positions = _hex_lattice_positions()
    centroid = positions.mean(axis=0)
    positions_centered = positions - centroid
    
    # Verify separation (should be guaranteed by lattice)
    assert _check_separation_fast(positions_centered), "Lattice positions must be separated"
    
    # Optimize outer hexagon: center (ox, oy) and angle
    # By symmetry, the optimal center should be near origin and angle near 0 or 30 degrees
    best_s = np.inf
    best_ox, best_oy, best_angle = 0.0, 0.0, 0.0
    
    # Try different angles and centers
    np.random.seed(42)
    
    # Grid search over angle first
    for angle_deg in np.arange(0, 60, 5):
        if time.time() > deadline - 30:
            break
        # For each angle, try a few center offsets
        for ox in np.linspace(-1.0, 1.0, 7):
            for oy in np.linspace(-1.0, 1.0, 7):
                if time.time() > deadline - 10:
                    break
                s = _min_containing_hex_side(positions_centered, ox, oy, angle_deg)
                if s < best_s:
                    best_s = s
                    best_ox, best_oy, best_angle = ox, oy, angle_deg
    
    # Refine with local search
    if time.time() < deadline - 30:
        ox, oy, angle = best_ox, best_oy, best_angle
        step = 0.5
        for it in range(200):
            if time.time() > deadline - 10:
                break
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for da in [-step, 0, step]:
                        if dx == 0 and dy == 0 and da == 0:
                            continue
                        new_ox, new_oy, new_angle = ox + dx, oy + dy, angle + da
                        s = _min_containing_hex_side(positions_centered, new_ox, new_oy, new_angle)
                        if s < best_s:
                            best_s = s
                            best_ox, best_oy, best_angle = new_ox, new_oy, new_angle
                            ox, oy, angle = new_ox, new_oy, new_angle
                            improved = True
            if not improved:
                step *= 0.5
                if step < 0.01:
                    break
    
    # Final result: shift inner positions by (-best_ox, -best_oy) so outer is at origin
    # Actually, the outer hexagon is at (best_ox, best_oy) with angle best_angle
    # We need to report: inner positions relative to outer center, outer at (0,0,angle)
    # Shift all inner positions by (-best_ox, -best_oy)
    inner_positions = positions_centered - np.array([best_ox, best_oy])
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = inner_positions[:, 0]
    inner_hex_data[:, 1] = inner_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
