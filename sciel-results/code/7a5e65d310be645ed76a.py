import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from itertools import product, combinations
from collections import deque

N = 12
SQRT3 = math.sqrt(3)
DEADLINE = 140.0

# Hexagonal lattice basis vectors
def hex_to_pixel(q, r):
    """Convert axial coordinates to pixel coordinates."""
    x = SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def pixel_to_hex(x, y):
    """Convert pixel coordinates to axial coordinates."""
    r = y / 1.5
    q = (x / SQRT3) - r / 2.0
    return q, r

def hex_round(q, r):
    """Round axial coordinates to nearest hexagon."""
    q, r = round(q), round(r)
    return q, r

def hex_neighbors(q, r):
    """Get all 6 neighbors of a hexagon at (q, r)."""
    return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

def get_hex_vertices(cx, cy, angle_rad, side=1.0):
    """Get vertices of a regular hexagon centered at (cx, cy) with given rotation."""
    verts = []
    for k in range(6):
        a = angle_rad + k * math.pi / 3
        verts.append([cx + side * math.cos(a), cy + side * math.sin(a)])
    return np.array(verts)

def polyhex_to_rigid_body(cells):
    """
    Convert a list of axial coordinates (polyhex) to a rigid body representation.
    Returns: list of (local_x, local_y) positions relative to centroid, and cell count.
    """
    pixels = [hex_to_pixel(q, r) for q, r in cells]
    cx = np.mean([p[0] for p in pixels])
    cy = np.mean([p[1] for p in pixels])
    local_positions = [(px - cx, py - cy) for px, py in pixels]
    return local_positions

def compute_min_enclosing_hexagon_side(local_positions, outer_angle_rad=0.0):
    """
    Compute minimum side length of a regular hexagon (with given rotation) 
    that can contain all unit hexagons at local_positions.
    Returns the side length.
    """
    # For each local position, compute the 6 vertices of a unit hexagon at that position
    # (all aligned with outer hexagon for now)
    all_verts = []
    for lx, ly in local_positions:
        verts = get_hex_vertices(lx, ly, 0.0, side=1.0)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    # Normals for the outer hexagon sides
    theta = outer_angle_rad
    normals = []
    for k in range(6):
        ang_n = theta + math.pi/6 + k * math.pi/3
        normals.append([math.cos(ang_n), math.sin(ang_n)])
    normals = np.array(normals)
    
    # Project all vertices onto each normal
    max_proj = 0.0
    for normal in normals:
        projs = all_verts @ normal
        max_proj = max(max_proj, np.max(projs))
    
    side = max_proj * 2.0 / SQRT3
    return side

def enumerate_polyhexes(n_cells, max_width=15):
    """
    Enumerate all connected polyhexes with n_cells cells using BFS growth.
    Returns list of polyhexes, each as a frozenset of (q, r) coordinates.
    """
    if n_cells == 1:
        return [frozenset([(0, 0)])]
    
    # Use a set to avoid duplicates
    seen = set()
    results = []
    
    # Start from all polyhexes with n_cells-1 cells
    prev_polyhexes = enumerate_polyhexes(n_cells - 1, max_width)
    
    for ph in prev_polyhexes:
        # Try adding a cell adjacent to each existing cell
        added_positions = set()
        for (q, r) in ph:
            for (nq, nr) in hex_neighbors(q, r):
                if (nq, nr) not in ph:
                    new_ph = frozenset(ph | {(nq, nr)})
                    # Canonicalize: translate so min q is 0, min r is 0
                    min_q = min(q for q, r in new_ph)
                    min_r = min(r for q, r in new_ph)
                    canon = frozenset((q - min_q, r - min_r) for q, r in new_ph)
                    
                    if canon not in seen:
                        seen.add(canon)
                        results.append(canon)
    
    return results

def canonicalize_polyhex(cells):
    """Canonicalize a polyhex by translating to origin and trying rotations/reflections."""
    # Translate so min q = 0, min r = 0
    min_q = min(q for q, r in cells)
    min_r = min(r for q, r in cells)
    translated = frozenset((q - min_q, r - min_r) for q, r in cells)
    return translated

def generate_initial_polyhexes(n=12):
    """
    Generate a diverse set of initial polyhex configurations.
    Returns list of frozensets of (q, r) coordinates.
    """
    configs = []
    
    # 1. Straight line
    configs.append(frozenset([(i, 0) for i in range(n)]))
    
    # 2. 2x6 rectangle (hexagonal tiling)
    for i in range(6):
        configs.append(frozenset([(i, j) for i in range(6) for j in range(2) if i*2+j < n]))
    
    # 3. Compact hexagonal shape (center + ring)
    center = (0, 0)
    ring1 = hex_neighbors(0, 0)
    ring2 = []
    for (q, r) in ring1:
        for (nq, nr) in hex_neighbors(q, r):
            if nq != 0 and nr != 0 and (nq, nr) not in ring1:
                if (nq, nr) not in ring2:
                    ring2.append((nq, nr))
    compact = frozenset([center] + ring1 + ring2[:n-7])
    configs.append(compact)
    
    # 4. Random connected shapes
    import random
    random.seed(42)
    for _ in range(20):
        cells = [(0, 0)]
        seen = {(0, 0)}
        for _ in range(n - 1):
            # Pick a random existing cell and add a random neighbor
            base = random.choice(cells)
            neighbors = [nb for nb in hex_neighbors(base[0], base[1]) if nb not in seen]
            if neighbors:
                new_cell = random.choice(neighbors)
                cells.append(new_cell)
                seen.add(new_cell)
        configs.append(frozenset(cells))
    
    # 5. Enumerate some actual polyhexes
    try:
        if time.time() < DEADLINE - 60:
            polyhexes = enumerate_polyhexes(n)
            # Take first 50 diverse ones
            for ph in polyhexes[:50]:
                configs.append(ph)
    except:
        pass
    
    # Remove duplicates
    unique_configs = []
    seen_canon = set()
    for cfg in configs:
        canon = canonicalize_polyhex(cfg)
        if canon not in seen_canon:
            seen_canon.add(canon)
            unique_configs.append(canon)
    
    return unique_configs

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    def calc_outer_side_for_rigid_body(local_positions, outer_center, outer_angle_rad):
        """
        Compute the outer hexagon side length for a rigid body at given position/rotation.
        local_positions: list of (lx, ly) relative to body centroid
        outer_center: (cx, cy) of outer hexagon
        outer_angle_rad: rotation of outer hexagon
        """
        all_verts = []
        for lx, ly in local_positions:
            # Rotate and translate the local position
            cos_a = math.cos(outer_angle_rad)
            sin_a = math.sin(outer_angle_rad)
            rx = lx * cos_a - ly * sin_a + outer_center[0]
            ry = lx * sin_a + ly * cos_a + outer_center[1]
            
            # Get vertices of unit hexagon at this position (aligned with outer)
            verts = get_hex_vertices(rx, ry, outer_angle_rad, side=1.0)
            all_verts.extend(verts)
        
        all_verts = np.array(all_verts)
        
        # Normals for the outer hexagon sides
        theta = outer_angle_rad
        normals = []
        for k in range(6):
            ang_n = theta + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        # Project all vertices onto each normal
        max_proj = 0.0
        for normal in normals:
            projs = all_verts @ normal
            max_proj = max(max_proj, np.max(projs))
        
        side = max_proj * 2.0 / SQRT3
        return side

    def optimize_rigid_body(local_positions):
        """
        Optimize global rotation and translation for a rigid body.
        Returns (best_side, best_translation, best_rotation).
        """
        def objective(params):
            tx, ty, angle = params
            return calc_outer_side_for_rigid_body(local_positions, (tx, ty), angle)
        
        # Try multiple initial angles
        best_side = float('inf')
        best_params = None
        
        for init_angle in [0, 30, 60, 90, 120, 150]:
            init_rad = math.radians(init_angle)
            x0 = np.array([0.0, 0.0, init_rad])
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x.copy()
            except:
                pass
        
        if best_params is None:
            return float('inf'), (0.0, 0.0), 0.0
        
        return best_side, (best_params[0], best_params[1]), best_params[2]

    # Generate initial polyhex configurations
    polyhex_configs = generate_initial_polyhexes(N)
    
    best_overall_side = float('inf')
    best_overall_config = None
    best_overall_translation = (0.0, 0.0)
    best_overall_rotation = 0.0

    # Optimize each polyhex configuration
    for idx, config in enumerate(polyhex_configs):
        if time.time() > deadline - 30:
            break
        
        local_positions = polyhex_to_rigid_body(list(config))
        
        # Compute the minimum enclosing hexagon side for this shape
        side, translation, rotation = optimize_rigid_body(local_positions)
        
        if side < best_overall_side:
            best_overall_side = side
            best_overall_config = config
            best_overall_translation = translation
            best_overall_rotation = rotation

    # If we didn't find a good config, use a fallback
    if best_overall_config is None:
        # Fallback: 3x4 grid of hexagons
        cells = []
        for i in range(3):
            for j in range(4):
                cells.append((j, i))
        best_overall_config = frozenset(cells)
        local_positions = polyhex_to_rigid_body(list(best_overall_config))
        best_overall_side, best_overall_translation, best_overall_rotation = optimize_rigid_body(local_positions)

    # Construct the final answer
    local_positions = polyhex_to_rigid_body(list(best_overall_config))
    tx, ty = best_overall_translation
    rot = best_overall_rotation
    
    cos_a = math.cos(rot)
    sin_a = math.sin(rot)
    
    inner = []
    for lx, ly in local_positions:
        # Rotate and translate
        rx = lx * cos_a - ly * sin_a + tx
        ry = lx * sin_a + ly * cos_a + ty
        inner.append([rx, ry, math.degrees(rot)])
    
    # Outer hexagon: center at (tx, ty), rotation = rot
    outer_center = (tx, ty)
    outer_side = best_overall_side
    outer_angle = math.degrees(rot)
    
    return inner, outer_center, outer_side, outer_angle
