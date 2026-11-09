# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (center, side, angle in deg)."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _separated(v1, v2, eps=1e-6):
    """Check if two convex hexagons (6 vertices each) are separated on at least one of the 12 edge normals."""
    for i in range(6):
        # Edge of v1
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Project v1 and v2 onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
            return True
        # Edge of v2
        q1 = v2[i]
        q2 = v2[(i + 1) % 6]
        edge2 = q2 - q1
        normal2 = np.array([-edge2[1], edge2[0]])
        proj1b = v1 @ normal2
        proj2b = v2 @ normal2
        if proj1b.max() < proj2b.min() - eps or proj2b.max() < proj1b.min() - eps:
            return True
    return False


def _all_separated(vertices_list, eps=1e-6):
    """Check all pairs of hexagons are separated."""
    n = len(vertices_list)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(vertices_list[i], vertices_list[j], eps):
                return False
    return True


def _minimal_enclosing_hexagon(inner_vertices_list, n_inner=11):
    """
    Find the minimal side length s and angle a such that all inner hexagon vertices
    are inside the outer regular hexagon centered at origin with side s and angle a.
    
    Uses linear programming: for a given angle a, the constraint that point p is inside
    hexagon (center 0, side s, angle a) is: for each of 6 edge normals n_k,
    p . n_k <= s * cos(30°) = s * sqrt(3)/2.
    
    So for fixed angle a, we need s >= max over all inner vertices p and all 6 normals n_k of (p . n_k) / (sqrt(3)/2).
    We minimize over angle a.
    
    Actually, the outer hexagon with side s and angle a has vertices at s*cos(a+60k), s*sin(a+60k).
    The edge from vertex k to k+1 has outward normal direction at angle a + 60k + 30°.
    The distance from center to each edge is s * cos(30°) = s * sqrt(3)/2.
    
    So constraint: for each inner vertex p and each k in 0..5:
        p . (cos(a+60k+30°), sin(a+60k+30°)) <= s * sqrt(3)/2
    
    For fixed a: s >= max_{p,k} (p . n_k(a)) / (sqrt(3)/2)
    We want to minimize s over a.
    
    This is a 1D optimization over a in [0, 60°) (periodicity).
    """
    # Collect all inner vertices
    all_pts = np.vstack(inner_vertices_list)
    n_pts = len(all_pts)
    
    def min_side_for_angle(a_deg):
        a = np.radians(a_deg)
        min_s = 0.0
        for k in range(6):
            n_k = np.array([np.cos(a + np.radians(60 * k + 30)), np.sin(a + np.radians(60 * k + 30))])
            proj = all_pts @ n_k
            max_proj = proj.max()
            s_needed = max_proj / (np.sqrt(3) / 2)
            if s_needed > min_s:
                min_s = s_needed
        return max(min_s, 0.0)
    
    # Search over angle in [0, 60) with fine resolution
    best_s = np.inf
    best_a = 0.0
    # Coarse search
    for a_deg in np.linspace(0, 60, 601, endpoint=False):
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    # Fine search around best
    for a_deg in np.linspace(best_a - 2, best_a + 2, 401):
        if a_deg < 0 or a_deg >= 60:
            a_deg = a_deg % 60
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg % 60
    
    return best_s, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length.
    """
    sqrt3 = np.sqrt(3)
    # Neighbor offsets for flat-top unit hexagons (angle 0)
    basis1 = np.array([1.5, sqrt3/2])
    basis2 = np.array([0, sqrt3])
    
    def axial_to_cartesian(q, r):
        return q * basis1 + r * basis2
    
    def check_valid_config(positions):
        """Check if all hexagons at given positions are mutually separated."""
        verts_list = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            verts_list.append(verts)
        return _all_separated(verts_list)
    
    def evaluate_config(positions):
        """Return (side_length, angle) for the minimal enclosing hexagon, or None if invalid."""
        if not check_valid_config(positions):
            return None
        verts_list = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            verts_list.append(verts)
        s, a = _minimal_enclosing_hexagon(verts_list)
        return (s, a)
    
    # Generate many candidate configurations systematically
    # Use axial coordinates. A hexagon at (q,r) is adjacent to (q±1,r), (q,r±1), (q+1,r-1), (q-1,r+1)
    # Two hexagons overlap if their axial distance is 0. We need |q1-q2| + |r1-r2| + |q1+r1-q2-r2| >= 2
    # Actually, in axial coords, the 6 neighbors of (q,r) are:
    # (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    
    best_side = np.inf
    best_positions = None
    best_angle = 0.0
    
    # Strategy: enumerate all connected sets of 11 hexagons in a bounded region
    # Use a greedy approach: start from center, add hexagons one by one to minimize bounding box
    
    # Precompute a pool of candidate axial coordinates
    max_c = 3
    all_axial = []
    for q in range(-max_c, max_c + 1):
        for r in range(-max_c, max_c + 1):
            all_axial.append((q, r))
    
    # Convert to set for fast lookup
    axial_set = set(all_axial)
    
    # Generate configurations by trying different "shapes" of 11-hexagon clusters
    # We'll use a systematic enumeration of subsets
    
    # For efficiency, let's try specific patterns:
    # Pattern 1: 3x3 block minus corners (7 hexes) + 4 more
    # Pattern 2: Hexagonal shape of radius 1 (7 hexes) + 4 more
    # Pattern 3: Various other compact shapes
    
    # Let's enumerate all subsets of size 11 from a smaller candidate set
    # and check which are valid (non-overlapping)
    
    # Use a smaller candidate set for combinatorial search
    small_max = 2
    small_axial = []
    for q in range(-small_max, small_max + 1):
        for r in range(-small_max, small_max + 1):
            small_axial.append((q, r))
    
    n_small = len(small_axial)
    # C(25, 11) is too large, so let's be smarter
    
    # Instead, let's generate configurations by starting from a core and adding hexagons
    # Core: center (0,0)
    # Then add neighbors in various orders
    
    # Define the 6 neighbors of (0,0)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Generate all configurations by choosing which 4 of the "next ring" to add
    # Ring 2: hexagons at distance 2 from center
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0:
                continue
            # Check if it's at hex distance 2
            # Hex distance in axial: max(|q|, |r|, |q+r|) ... actually it's (|q| + |r| + |q+r|)/2
            dist = (abs(q) + abs(r) + abs(q+r)) // 2
            if dist == 2:
                ring2.append((q, r))
    
    # Also include some ring 1 hexagons that weren't in the core
    # Actually let's just try many explicit configurations
    
    configs_to_try = []
    
    # Config: center + all 6 neighbors + 4 from ring2
    for combo in itertools.combinations(ring2, 4):
        config = [(0,0)] + neighbors + list(combo)
        configs_to_try.append(config)
    
    # Config: center + 5 neighbors + 5 from ring2
    for combo5 in itertools.combinations(neighbors, 5):
        for combo4 in itertools.combinations(ring2, 5):
            config = [(0,0)] + list(combo5) + list(combo4)
            configs_to_try.append(config)
    
    # Config: center + 4 neighbors + 6 from ring2
    for combo4 in itertools.combinations(neighbors, 4):
        for combo6 in itertools.combinations(ring2, 6):
            config = [(0,0)] + list(combo4) + list(combo6)
            configs_to_try.append(config)
    
    # Config: no center, 11 from neighbors + ring2
    for combo11 in itertools.combinations(neighbors + ring2, 11):
        configs_to_try.append(list(combo11))
    
    # Also try some manually designed configurations
    manual_configs = [
        # 3-3-3-2 pattern
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,-1), (1,-1), (2,-1), (-1,1), (-1,0)],
        # Compact hexagon shape
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)],
        # Another variant
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (1,2), (-1,2), (-2,1)],
    ]
    configs_to_try.extend(manual_configs)
    
    # Deduplicate
    seen = set()
    unique_configs = []
    for config in configs_to_try:
        key = tuple(sorted(config))
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    # Evaluate each configuration
    for config in unique_configs:
        positions = []
        for (q, r) in config:
            pos = axial_to_cartesian(q, r)
            positions.append(pos)
        
        result = evaluate_config(positions)
        if result is not None:
            s, a = result
            if s < best_side:
                best_side = s
                best_positions = positions
                best_angle = a
    
    if best_positions is None:
        # Fallback
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Build inner hex data
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(best_positions):
        inner_hex_data[i] = [pos[0], pos[1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
