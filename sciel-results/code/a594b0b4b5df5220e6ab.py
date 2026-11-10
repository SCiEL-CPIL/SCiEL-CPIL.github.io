# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


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


def _minimal_enclosing_hexagon(all_pts):
    """
    Find the minimal side length s and angle a such that all points are inside
    the outer regular hexagon centered at origin with side s and angle a.
    """
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
    
    best_s = np.inf
    best_a = 0.0
    
    # Coarse search
    coarse_angles = np.linspace(0, 60, 361, endpoint=False)
    for a_deg in coarse_angles:
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    # Refine with multiple local searches
    candidates = []
    for a_deg in coarse_angles:
        s = min_side_for_angle(a_deg)
        if s < best_s + 0.01:
            candidates.append(a_deg)
    
    # Group nearby candidates and refine each group
    if candidates:
        candidates.sort()
        groups = []
        current_group = [candidates[0]]
        for c in candidates[1:]:
            if c - current_group[-1] < 2.0:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]
        groups.append(current_group)
        
        for group in groups:
            a_center = np.mean(group)
            a_lower = a_center - 2.0
            a_upper = a_center + 2.0
            
            def objective(a_deg):
                return min_side_for_angle(a_deg)
            
            res = minimize_scalar(objective, bounds=(a_lower, a_upper), method='bounded', options={'xatol': 1e-10})
            if res.success and res.fun < best_s:
                best_s = res.fun
                best_a = res.x
    
    best_a = best_a % 60
    return best_s, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length.
    """
    sqrt3 = np.sqrt(3)
    offsets = np.array([
        [1.5, sqrt3/2],
        [0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0, -sqrt3],
        [1.5, -sqrt3/2]
    ])
    
    basis1 = offsets[0]
    basis2 = offsets[1]
    
    # Enumerate many compact shapes in axial coordinates
    # Center (0,0) + 6 neighbors + 4 additional
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Additional candidates at distance 2 from center:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # Also (1,1), (-1,-1) are at distance sqrt(3) but not neighbors
    
    base_seven = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Distance-2 positions (hex distance 2)
    dist2_positions = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    
    # Also try positions like (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)
    other_positions = [(1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2), (3,0), (-3,0), (0,3), (0,-3), (-3,3), (3,-3)]
    
    shapes = []
    seen_shapes = set()
    
    # Try all combinations of 4 from dist2_positions
    from itertools import combinations
    for combo in combinations(dist2_positions, 4):
        shape = base_seven + list(combo)
        if len(set(shape)) == 11:
            key = tuple(sorted(shape))
            if key not in seen_shapes:
                seen_shapes.add(key)
                shapes.append(shape)
    
    # Try combinations mixing dist2 and other positions
    all_extra = dist2_positions + other_positions
    # Limit to avoid explosion: try 4 from a subset
    # First 8 most promising
    promising_extra = dist2_positions + [(1,1), (-1,-1), (2,-1), (-2,1)]
    for combo in combinations(promising_extra, 4):
        shape = base_seven + list(combo)
        if len(set(shape)) == 11:
            key = tuple(sorted(shape))
            if key not in seen_shapes:
                seen_shapes.add(key)
                shapes.append(shape)
    
    # Also try shapes that don't include all 6 neighbors
    # For example, center + 5 neighbors + 5 others
    # But this is more complex; skip for now
    
    best_result = None
    best_side = np.inf
    
    # Small expansion to ensure strict separation
    lambda_val = 1.0 + 1e-5
    
    for shape in shapes:
        base_positions = []
        for (q, r) in shape:
            pos = q * basis1 + r * basis2
            base_positions.append(pos)
        base_positions = np.array(base_positions)
        
        positions = base_positions * lambda_val
        
        # Compute vertices
        verts_list = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            verts_list.append(verts)
        
        # Check separation
        if not _all_separated(verts_list):
            continue
        
        # Compute minimal enclosing hexagon
        all_pts = np.vstack(verts_list)
        s, a = _minimal_enclosing_hexagon(all_pts)
        
        if s < best_side:
            best_side = s
            best_result = (positions, s, a)
    
    if best_result is None:
        # Fallback with guaranteed separation
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
    
    positions, best_s, best_a = best_result
    
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i] = [pos[0], pos[1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
