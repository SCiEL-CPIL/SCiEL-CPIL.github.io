import numpy as np
import itertools

N = 12

def get_hex_vertices(center, angle_deg=0):
    """
    Returns vertices of a unit regular hexagon centered at 'center' with rotation 'angle_deg'.
    Standard orientation: flat top (vertex at 0 deg? No, flat top means side is horizontal at top).
    Let's assume the standard orientation used in the seed.
    Seed: inner = [[4.0 * k, 0.0, 0.0] ...]
    If angle is 0, and it's flat top, vertices are at angles 0, 60, 120...?
    If vertices are at 0, 60..., the top is a vertex (pointy top).
    If sides are horizontal, vertices are at 30, 90, 150...?
    Let's check the distance.
    If pointy top (vertex at 0), width is 2 (from -1 to 1), height is sqrt(3).
    If flat top (side at top), width is sqrt(3), height is 2.
    The seed uses x = 4k. Spacing 4.
    If pointy top, width 2. Gap 2.
    If flat top, width sqrt(3) ~ 1.732. Gap 2.268.
    Usually, "regular hexagon" without spec might imply vertices on axes?
    Let's assume vertices at angles $k \cdot 60^\circ$.
    Vertex 0: $(1, 0)$.
    Vertex 1: $(0.5, \sqrt{3}/2)$.
    This is "pointy top" if y is up? No, vertex at right is 0 deg.
    Top vertex is at 90 deg? No, 60 deg is $(0.5, 0.866)$. 120 deg is $(-0.5, 0.866)$.
    So the top is a flat edge connecting 60 and 120 deg vertices.
    So this is "flat top" in the sense that the top boundary is a horizontal line segment.
    Yes, vertices at 0, 60, 120, 180, 240, 300.
    Top edge is between 60 and 120. y = sqrt(3)/2. Horizontal.
    So this matches my outer hexagon definition (flat top).
    """
    cx, cy = center
    rad = np.deg2rad(angle_deg)
    cos_r = np.cos(rad)
    sin_r = np.sin(rad)
    
    # Base vertices for unit hexagon, rotation 0
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    
    # Rotate
    rot_x = base_x * cos_r - base_y * sin_r
    rot_y = base_x * sin_r + base_y * cos_r
    
    # Translate
    verts_x = rot_x + cx
    verts_y = rot_y + cy
    
    return np.column_stack((verts_x, verts_y))

def min_outer_side(vertices):
    """
    Calculate minimum side length L of a regular hexagon (centered at 0,0, rotation 0)
    that contains all points in 'vertices'.
    """
    if len(vertices) == 0:
        return 0.0
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Inequalities for point (x,y) to be in hexagon of side L:
    # 1. |x| <= L
    # 2. |y| <= sqrt(3)/2 * L  => L >= 2/sqrt(3) * |y|
    # 3. sqrt(3)*x + y <= sqrt(3)*L => L >= (sqrt(3)*x + y)/sqrt(3)
    # 4. -sqrt(3)*x + y <= sqrt(3)*L => L >= (-sqrt(3)*x + y)/sqrt(3)
    # 5. sqrt(3)*x - y <= sqrt(3)*L => L >= (sqrt(3)*x - y)/sqrt(3)
    # 6. -sqrt(3)*x - y <= sqrt(3)*L => L >= (-sqrt(3)*x - y)/sqrt(3)
    
    # Note: The inequalities must hold for ALL x, y.
    # So L >= max( |x|, 2/sqrt(3)|y|, ... )
    
    c1 = np.max(np.abs(x))
    c2 = np.max(np.abs(y)) * (2.0 / np.sqrt(3.0))
    
    # For the slanted ones, we need max of the linear forms.
    # Since L must be positive, we take max of the positive parts?
    # Actually, the condition is sqrt(3)x + y <= sqrt(3)L.
    # So L >= (sqrt(3)x + y)/sqrt(3).
    # We need this to hold for all points.
    # So L >= max( (sqrt(3)x + y)/sqrt(3) for all points ).
    
    c3 = np.max(np.sqrt(3.0)*x + y) / np.sqrt(3.0)
    c4 = np.max(-np.sqrt(3.0)*x + y) / np.sqrt(3.0)
    c5 = np.max(np.sqrt(3.0)*x - y) / np.sqrt(3.0)
    c6 = np.max(-np.sqrt(3.0)*x - y) / np.sqrt(3.0)
    
    return max(c1, c2, c3, c4, c5, c6)

def construct():
    # Generate candidate centers using triangular lattice
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3)*(q + r/2), y = 3/2*r
    
    # We need 12 centers.
    # Let's try a few clusters.
    
    # Cluster 1: 3x4 rectangle in lattice?
    # Cluster 2: Hexagonal-ish 12
    
    # Let's define a function to evaluate a set of axial coords
    def eval_cluster(axial_coords):
        centers = []
        for q, r in axial_coords:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            centers.append((x, y))
        
        all_verts = []
        for c in centers:
            v = get_hex_vertices(c, 0)
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)
        return min_outer_side(all_verts)
        
    # Candidate A: Compact 12
    # Center (0,0)
    # Ring 1 (6): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (12): (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1)
    
    # We pick 5 from Ring 2.
    # To be symmetric, maybe pick every other one?
    # Or just try all combinations of 5 from 12? C(12,5) = 792. Fast.
    
    base = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1)]
    
    best_L = float('inf')
    best_config = None
    
    # Try all subsets of 5 from ring 2
    for subset in itertools.combinations(ring2, 5):
        coords = base + ring1 + list(subset)
        L = eval_cluster(coords)
        if L < best_L:
            best_L = L
            best_config = coords
            
    # Also try other shapes? E.g. 4x3 grid
    # Grid: q in [-1, 2], r in [0, 3]?
    # Let's just stick to the best found in the "flower" shape.
    # It's likely optimal or very close.
    
    # Convert best_config to cartesian and return
    inner = []
    for q, r in best_config:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import itertools
    
    N = 12
    
    def get_hex_vertices(center, angle_deg=0):
        cx, cy = center
        rad = np.deg2rad(angle_deg)
        cos_r = np.cos(rad)
        sin_r = np.sin(rad)
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        base_x = np.cos(angles)
        base_y = np.sin(angles)
        rot_x = base_x * cos_r - base_y * sin_r
        rot_y = base_x * sin_r + base_y * cos_r
        verts_x = rot_x + cx
        verts_y = rot_y + cy
        return np.column_stack((verts_x, verts_y))

    def min_outer_side(vertices):
        if len(vertices) == 0:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        c1 = np.max(np.abs(x))
        c2 = np.max(np.abs(y)) * (2.0 / np.sqrt(3.0))
        c3 = np.max(np.sqrt(3.0)*x + y) / np.sqrt(3.0)
        c4 = np.max(-np.sqrt(3.0)*x + y) / np.sqrt(3.0)
        c5 = np.max(np.sqrt(3.0)*x - y) / np.sqrt(3.0)
        c6 = np.max(-np.sqrt(3.0)*x - y) / np.sqrt(3.0)
        return max(c1, c2, c3, c4, c5, c6)

    def eval_cluster(axial_coords):
        centers = []
        for q, r in axial_coords:
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            centers.append((x, y))
        
        all_verts = []
        for c in centers:
            v = get_hex_vertices(c, 0)
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)
        return min_outer_side(all_verts)

    base = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1)]
    
    best_L = float('inf')
    best_config = None
    
    for subset in itertools.combinations(ring2, 5):
        coords = base + ring1 + list(subset)
        L = eval_cluster(coords)
        if L < best_L:
            best_L = L
            best_config = coords
            
    inner = []
    for q, r in best_config:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
