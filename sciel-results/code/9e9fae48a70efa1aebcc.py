"""Unit regular hexagons inside a regular hexagon n=12."""
import math
import time

N = 12

# --- Geometry helpers ---

def hex_vertices(cx, cy, side, angle_rad):
    """Return list of 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_rad.
    Vertices are in CCW order starting from angle 0 (rightmost vertex when angle=0)."""
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def hex_to_point_dist(px, py, cx, cy, side, angle_rad):
    """Chebyshev-like distance: max over the 6 support lines of the hexagon.
    Actually we need the minimum side S such that point (px,py) is inside the hexagon.
    For a regular hexagon centered at (cx,cy) with side S and rotation angle,
    the hexagon is the intersection of 6 half-planes.
    For each edge, the distance from center to edge (apothem) is S * sqrt(3)/2.
    A point is inside iff for all 6 normals n_i, dot((px-cx,py-cy), n_i) <= S*sqrt(3)/2.
    So S >= 2/sqrt(3) * max_i dot((px-cx,py-cy), n_i).
    """
    dx = px - cx
    dy = py - cy
    max_proj = 0.0
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        # normal to edge i points outward
        nx = math.cos(a + math.pi/2)  # wait, let me reconsider
        # Actually for a regular hexagon, the outward normal of edge between vertices i and i+1
        # is in the direction of the midpoint of that edge.
        # The midpoint of edge i (between vertex i and i+1) is at angle angle_rad + (i+0.5)*pi/3
        # The outward normal is in that same direction.
        na = angle_rad + (i + 0.5) * math.pi / 3.0
        nx = math.cos(na)
        ny = math.sin(na)
        proj = dx * nx + dy * ny
        if proj > max_proj:
            max_proj = proj
    # S * sqrt(3)/2 >= max_proj => S >= 2*max_proj/sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def min_side_for_hexagons(positions, outer_angle_rad, outer_center):
    """Given positions of unit hexagons (centers), and outer hexagon center and angle,
    find minimum outer side length S such that all unit hexagons fit inside.
    Each unit hexagon has side 1. Its vertices are at distance 1 from its center.
    For the unit hexagon at (px,py) to be inside outer hexagon (center oc, side S, angle oa),
    all 6 vertices of the inner hexagon must satisfy the half-plane constraints of the outer hex.
    Equivalently, the inner hexagon's center must be inside the outer hexagon eroded by the
    inner hexagon. The Minkowski difference is another hexagon.
    Actually, let's just compute: for each inner hexagon vertex, it must be inside the outer hex.
    For vertex v of inner hex at (px,py) with angle ia=0 (unit hex, we can assume same orientation for simplicity... 
    but the task allows rotation. Let's assume all inner hexes have same rotation as outer for now, or 0.
    Actually the inner hexes can rotate independently. But for optimal packing on a hex lattice,
    they all have the same orientation. Let's assume all inner hexes have angle 0 for now.
    """
    ocx, ocy = outer_center
    S = 0.0
    for (px, py) in positions:
        # Inner hex vertices (side=1, angle=0)
        for i in range(6):
            a = i * math.pi / 3.0
            vx = px + math.cos(a)
            vy = py + math.sin(a)
            # Distance from outer center
            dx = vx - ocx
            dy = vy - ocy
            max_proj = 0.0
            for j in range(6):
                na = outer_angle_rad + (j + 0.5) * math.pi / 3.0
                proj = dx * math.cos(na) + dy * math.sin(na)
                if proj > max_proj:
                    max_proj = proj
            needed = 2.0 * max_proj / math.sqrt(3.0)
            if needed > S:
                S = needed
    return S

# --- Generate hex lattice arrangements for 12 cells ---

def generate_hex_lattice_arrangements(n):
    """Generate all connected subsets of size n from the hexagonal lattice.
    We use axial coordinates. Each cell is a unit hexagon.
    Adjacent cells have centers at distance sqrt(3) (for side-1 hexagons).
    Actually, for regular hexagons of side 1 tiled in a honeycomb, 
    the center-to-center distance between adjacent hexagons is 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3).
    Wait: apothem of unit hexagon = sqrt(3)/2 * side = sqrt(3)/2.
    Distance between centers of adjacent hexes in a tiling = 2 * apothem = sqrt(3).
    
    We use axial coordinates (q, r). The 6 neighbors are:
    (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    The Cartesian position of hex center at axial (q,r) with side s:
    x = s * 3/2 * q
    y = s * sqrt(3) * (r + q/2)
    
    For unit hexagons (s=1): x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    We want to enumerate all connected polyhexes of size n.
    For n=12, this is a large number. Let's use a systematic approach:
    BFS/DFS growing the polyhex, using symmetry to reduce.
    """
    sqrt3 = math.sqrt(3)
    directions = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    
    # Canonical representation: translate so min-q is 0, then min-r among min-q is 0
    def canonical(cells):
        cells = sorted(cells)
        min_q = min(q for q,r in cells)
        shifted = [(q-min_q, r) for q,r in cells]
        min_r = min(r for q,r in shifted if q == 0)
        shifted2 = [(q, r-min_r) for q,r in shifted]
        return tuple(sorted(shifted2))
    
    # Use a set to track seen canonical forms
    seen = set()
    results = []
    
    # Start with a single cell at (0,0)
    initial = [(0,0)]
    queue = [initial]
    seen.add(canonical(initial))
    
    while queue:
        cells = queue.pop(0)
        if len(cells) == n:
            # Convert to Cartesian
            cart = []
            for q,r in cells:
                x = 1.5 * q
                y = sqrt3 * (r + q/2.0)
                cart.append((x, y))
            results.append(cart)
            continue
        # Try adding each neighbor of each existing cell
        cell_set = set(cells)
        for q,r in cells:
            for dq,dr in directions:
                nq, nr = q+dq, r+dr
                if (nq, nr) in cell_set:
                    continue
                new_cells = cells + [(nq, nr)]
                can = canonical(new_cells)
                if can not in seen:
                    seen.add(can)
                    queue.append(new_cells)
    
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    # For n=12, the number of free polyhexes is 65245 (known value).
    # Enumerating all and optimizing each may be too slow.
    # Let's try a smarter approach: use known compact shapes.
    # The most compact shapes are those close to a larger hexagon.
    # A hexagon of side k has 3k(k-1)+1 cells.
    # k=1: 1, k=2: 7, k=3: 19. So 12 is between 7 and 19.
    # A "2x2" block plus some, or a compact arrangement.
    
    # Let's generate a few promising arrangements and optimize them.
    # Approach: Start with the 7-cell hexagon (side 2) and add 5 more cells 
    # in the most compact way.
    
    sqrt3 = math.sqrt(3)
    
    # The 7-cell hexagon: (0,0) and its 6 neighbors
    base7 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    
    # Add 5 more cells to make 12. Try different additions.
    # We'll use a heuristic: grow the polyhex to minimize bounding hexagon.
    
    # Let's just do a limited search: enumerate polyhexes but stop early.
    # Actually, for n=12 there are 65245 free polyhexes. That's too many to all optimize.
    
    # Alternative: use a genetic algorithm or simulated annealing on continuous positions.
    # But the directive says to try hexagonal lattice topologies.
    
    # Let's try a different approach: assume all inner hexagons are on a hex lattice
    # with the same orientation as the outer. The lattice is fixed. We just need to find
    # which 12 lattice points to use and what outer hexagon (center, side, angle) minimizes S.
    
    # For a given set of 12 lattice points, the optimal outer hexagon can be found by
    # trying different angles and computing the minimum enclosing hexagon.
    
    # Let's generate a candidate set of arrangements using a simple growth heuristic.
    
    arrangements = []
    
    # Strategy: grow from the 7-cell hexagon by adding cells one by one,
    # choosing the cell that minimizes the increase in bounding hexagon side.
    
    def to_cartesian(cells):
        return [(1.5*q, sqrt3*(r + q/2.0)) for q,r in cells]
    
    def bounding_hex_side(cart, angle_rad):
        """Find minimum side of regular hexagon with given angle that contains all points."""
        if not cart:
            return 0.0
        # Try to center at centroid
        cx = sum(p[0] for p in cart) / len(cart)
        cy = sum(p[1] for p in cart) / len(cart)
        return min_side_for_hexagons(cart, angle_rad, (cx, cy))
    
    # Grow from base7
    current = list(base7)
    cell_set = set(current)
    directions = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    
    best_arr = None
    best_side = float('inf')
    
    # Try different growth orders using a greedy approach with randomization
    import random
    random.seed(42)
    
    for trial in range(20):
        cur = list(base7)
        cur_set = set(cur)
        
        for _ in range(5):
            # Find all valid additions
            candidates = []
            for q,r in cur:
                for dq,dr in directions:
                    nq, nr = q+dq, r+dr
                    if (nq,nr) not in cur_set:
                        candidates.append((nq,nr))
            if not candidates:
                break
            
            # For each candidate, compute the bounding hex side if added
            best_cand = None
            best_cand_side = float('inf')
            for cand in candidates:
                new_cur = cur + [cand]
                cart = to_cartesian(new_cur)
                # Try a few angles
                for ang in [0, math.pi/6, math.pi/3, math.pi/2]:
                    side = bounding_hex_side(cart, ang)
                    if side < best_cand_side:
                        best_cand_side = side
                        best_cand = cand
            
            if best_cand is None:
                break
            cur.append(best_cand)
            cur_set.add(best_cand)
        
        if len(cur) == 12:
            cart = to_cartesian(cur)
            # Optimize angle more carefully
            best_angle = 0.0
            best_s = float('inf')
            for ang_deg in range(0, 60, 2):
                ang = math.radians(ang_deg)
                s = bounding_hex_side(cart, ang)
                if s < best_s:
                    best_s = s
                    best_angle = ang
            
            # Refine angle
            ang_lo = best_angle - math.radians(2)
            ang_hi = best_angle + math.radians(2)
            for _ in range(20):
                mid1 = ang_lo + (ang_hi - ang_lo) / 3.0
                mid2 = ang_hi - (ang_hi - ang_lo) / 3.0
                s1 = bounding_hex_side(cart, mid1)
                s2 = bounding_hex_side(cart, mid2)
                if s1 < s2:
                    ang_hi = mid2
                else:
                    ang_lo = mid1
                best_angle = (ang_lo + ang_hi) / 2
                best_s = bounding_hex_side(cart, best_angle)
            
            if best_s < best_side:
                best_side = best_s
                best_arr = cart
                best_final_angle = best_angle
            
            if time.time() > deadline:
                break
    
    if best_arr is None or not math.isfinite(best_side):
        # Fallback: line arrangement
        best_arr = [(k * sqrt3, 0.0) for k in range(12)]
        best_final_angle = 0.0
        best_side = bounding_hex_side(best_arr, 0.0)
    
    # Find optimal center for the best arrangement
    cart = best_arr
    cx = sum(p[0] for p in cart) / len(cart)
    cy = sum(p[1] for p in cart) / len(cart)
    best_center = (cx, cy)
    
    # Final optimization of center and angle together (simple grid)
    best_total = float('inf')
    best_cx, best_cy = cx, cy
    best_ang = best_final_angle
    
    # The center should be the centroid for symmetric arrangements
    # Just use centroid
    best_cx = sum(p[0] for p in cart) / len(cart)
    best_cy = sum(p[1] for p in cart) / len(cart)
    best_ang = best_final_angle
    best_total = min_side_for_hexagons(cart, best_ang, (best_cx, best_cy))
    
    inner = []
    for (px, py) in cart:
        inner.append([px, py, math.degrees(best_ang)])
    
    return inner, (best_cx, best_cy), best_total, math.degrees(best_ang)
