"""Packing 11 unit hexagons in a regular hexagon using cluster-based GA."""
import numpy as np
from scipy.spatial.distance import cdist

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _hex_support(cx, cy, angle_deg, u, v):
    """Support function h(u,v) of unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    # Unit hexagon support function in direction (u,v)
    ca, sa = np.cos(a), np.sin(a)
    # Rotate direction into hexagon's local frame
    u_local = u * ca + v * sa
    v_local = -u * sa + v * ca
    # Support of regular hexagon with flat-top orientation (vertices at 0,60,...)
    # For a regular hexagon with circumradius 1, the support in direction (dx,dy) is:
    # max over 6 vertices of dot(vertex, dir)
    # Vertex angles: 0, 60, 120, 180, 240, 300 degrees
    # Support = max_i cos(theta_i - phi) where phi = atan2(dy,dx)
    # Simpler: compute max of dot products
    verts = np.column_stack([np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)])
    return max(np.dot(verts, [u_local, v_local]))

def _outer_hex_support(cx, cy, side, angle_deg, u, v):
    """Support function of outer regular hexagon centered at (cx,cy), side length, rotated."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    u_local = u * ca + v * sa
    v_local = -u * sa + v * ca
    # Outer hexagon: regular hexagon with side length = side, so circumradius = side
    # Vertices at angles 0, 60, ... with radius = side
    verts = side * np.column_stack([np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)])
    return max(np.dot(verts, [u_local, v_local]))

def _min_dist_hex_hex(cx1, cy1, a1, cx2, cy2, a2):
    """Minimum distance between two unit hexagons (positive if disjoint, negative if overlap).
    Using support functions: dist = min over directions of (h1(u) + h2(-u) - dot(center_diff, u))
    Actually: min_dist = min_u [h1(u) - h2(u) - dot(c1-c2, u)] ... let me use the standard formula.
    Two convex bodies A, B: min dist = min_u [h_A(u) + h_B(-u) - dot(cB - cA, u)]
    where h is support function. For disjoint, this is > 0.
    """
    dx, dy = cx2 - cx1, cy2 - cy1
    # Search over 12 directions (6 from each hexagon's normals)
    best = float('inf')
    dirs = []
    for ang in [a1, a2]:
        a = np.radians(ang)
        for k in range(6):
            theta = a + k * np.pi / 3.0 + np.pi / 6.0  # edge normals
            dirs.append((np.cos(theta), np.sin(theta)))
    for u, v in dirs:
        h1 = _hex_support(cx1, cy1, a1, u, v)
        h2 = _hex_support(cx2, cy2, a2, -u, -v)
        val = h1 + h2 - (u * dx + v * dy)
        if val < best:
            best = val
    return best

def _max_violation_inner(inner):
    """Return max overlap violation among inner hexagons (positive means overlap)."""
    n = len(inner)
    max_v = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            d = _min_dist_hex_hex(x1, y1, a1, x2, y2, a2)
            if d < 0:
                max_v = max(max_v, -d)
    return max_v

def _min_dist_to_outer(inner, outer_cx, outer_cy, outer_side, outer_angle):
    """Return max violation of inner hexagons going outside outer hexagon."""
    max_v = 0.0
    for x, y, a in inner:
        # Check all 6 edges of outer hexagon
        oa = np.radians(outer_angle)
        for k in range(6):
            theta = oa + k * np.pi / 3.0 + np.pi / 6.0
            u, v = np.cos(theta), np.sin(theta)
            h_outer = _outer_hex_support(outer_cx, outer_cy, outer_side, outer_angle, u, v)
            h_inner = _hex_support(x, y, a, u, v)
            # Distance from outer edge to inner hexagon in direction (u,v)
            # If inner hexagon extends beyond outer, violation
            c_diff_u = u * (x - outer_cx) + v * (y - outer_cy)
            dist = h_outer - h_inner - c_diff_u
            if dist < 0:
                max_v = max(max_v, -dist)
    return max_v

def _evaluate(inner, outer_cx, outer_cy, outer_side, outer_angle):
    """Evaluate a configuration. Return (penalty, outer_side) where penalty is max violation."""
    inner_viol = _max_violation_inner(inner)
    outer_viol = _min_dist_to_outer(inner, outer_cx, outer_cy, outer_side, outer_angle)
    penalty = max(inner_viol, outer_viol)
    return penalty, outer_side

def _compute_min_side(inner, outer_cx, outer_cy, outer_angle):
    """Compute minimum outer side length needed to contain all inner hexagons."""
    # For each direction (6 edge normals of outer hex), find required side
    max_needed = 0.0
    oa = np.radians(outer_angle)
    for k in range(6):
        theta = oa + k * np.pi / 3.0 + np.pi / 6.0
        u, v = np.cos(theta), np.sin(theta)
        # Max extent of inner hexagons in direction (u,v)
        max_ext = 0.0
        for x, y, a in inner:
            h_inner = _hex_support(x, y, a, u, v)
            c_diff = u * x + v * y
            ext = h_inner + c_diff
            if ext > max_ext:
                max_ext = ext
        # Outer hexagon support in same direction must be >= max_ext
        # h_outer = side * support_unit(u,v) where support_unit is for unit hexagon
        # side * s >= max_ext => side >= max_ext / s
        # Compute support of unit hexagon in direction (u,v) rotated by outer_angle
        ca, sa = np.cos(oa), np.sin(oa)
        u_local = u * ca + v * sa
        v_local = -u * sa + v * ca
        verts = np.column_stack([np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)])
        s = max(np.dot(verts, [u_local, v_local]))
        if s > 1e-9:
            needed = max_ext / s
            if needed > max_needed:
                max_needed = needed
    return max_needed

def construct():
    """
    Use a cluster-based approach: divide 11 hexagons into clusters,
    optimize cluster positions/rotations with a GA, then fine-tune.
    """
    np.random.seed(42)
    
    # Initialize with a reasonable hexagonal packing pattern
    # 11 hexagons: center + 6 around + 4 more
    def init_config():
        # Try a few initial patterns
        patterns = []
        
        # Pattern 1: Center + ring of 6 + 4 extra
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            a = k * np.pi / 3.0
            inner.append([2.0 * np.cos(a), 2.0 * np.sin(a), 0.0])
        # Add 4 more at outer ring positions
        for k in [0, 2, 4, 1]:
            a = k * np.pi / 3.0 + np.pi / 6.0
            r = 2.0 * np.sqrt(3)
            inner.append([r * np.cos(a), r * np.sin(a), 0.0])
        patterns.append(inner)
        
        # Pattern 2: Linear-ish arrangement
        inner = []
        for i in range(6):
            inner.append([2.0 * i, 0.0, 0.0])
        for i in range(5):
            inner.append([2.0 * i + 1.0, 1.732, 0.0])
        patterns.append(inner)
        
        # Pattern 3: Compact cluster
        inner = [[0.0, 0.0, 0.0]]
        positions = [
            (2, 0), (-2, 0), (1, 1.732), (-1, 1.732), (1, -1.732), (-1, -1.732),
            (3, 1.732), (-3, 1.732), (3, -1.732), (-3, -1.732)
        ]
        for x, y in positions:
            inner.append([float(x), float(y), 0.0])
        patterns.append(inner)
        
        return patterns
    
    patterns = init_config()
    
    # Evaluate each pattern to find best starting point
    best_pattern = None
    best_side = float('inf')
    for pat in patterns:
        side = _compute_min_side(pat, 0.0, 0.0, 0.0)
        # Check validity
        viol = _max_violation_inner(pat)
        if viol < 1e-6:
            if side < best_side:
                best_side = side
                best_pattern = [list(p) for p in pat]
    
    if best_pattern is None:
        best_pattern = [list(p) for p in patterns[0]]
        best_side = _compute_min_side(best_pattern, 0.0, 0.0, 0.0)
    
    # GA to optimize: perturb positions and angles
    def fitness(inner):
        side = _compute_min_side(inner, 0.0, 0.0, 0.0)
        viol = _max_violation_inner(inner)
        if viol > 1e-4:
            return side + 100.0 * viol
        return side
    
    current = [list(p) for p in best_pattern]
    current_fit = fitness(current)
    
    # Simulated annealing / GA hybrid
    T = 5.0
    T_min = 0.001
    cooling = 0.9995
    max_iter = 8000
    
    for it in range(max_iter):
        if T < T_min:
            break
        # Perturb a random subset
        candidate = [list(p) for p in current]
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            da = np.random.normal(0, 5.0)
            candidate[idx][0] += dx
            candidate[idx][1] += dy
            candidate[idx][2] += da
        
        cand_fit = fitness(candidate)
        
        # Accept if better or with probability exp(-delta/T)
        if cand_fit < current_fit:
            current = candidate
            current_fit = cand_fit
        else:
            delta = cand_fit - current_fit
            if delta < 0:
                delta = 0
            prob = np.exp(-delta / T)
            if np.random.random() < prob:
                current = candidate
                current_fit = cand_fit
        
        T *= cooling
    
    # Final: compute optimal outer side and center
    side = _compute_min_side(current, 0.0, 0.0, 0.0)
    
    # Center the configuration optimally
    # Compute bounding box in all 6 directions to center
    # For simplicity, center at origin and use computed side
    return current, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
