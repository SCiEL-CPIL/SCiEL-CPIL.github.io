import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_polygon(px, py, verts):
    """Ray-casting point in polygon. verts shape (6,2)."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _polygons_overlap(v1, v2, eps=1e-9):
    """Return True if two convex hexagons (each shape (6,2)) overlap (including boundary).
    Uses SAT: if any axis (normal to an edge) separates them, no overlap."""
    def project(verts, axis):
        return np.dot(verts, axis)
    
    axes = []
    for i in range(6):
        edge = v1[i + 1] - v1[i]
        axes.append(np.array([-edge[1], edge[0]]))
        edge = v2[i + 1] - v2[i]
        axes.append(np.array([-edge[1], edge[0]]))
    
    for ax in axes:
        ax_norm = np.linalg.norm(ax)
        if ax_norm < 1e-12:
            continue
        ax = ax / ax_norm
        p1 = project(v1, ax)
        p2 = project(v2, ax)
        if p1.max() + eps < p2.min() or p2.max() + eps < p1.min():
            return False
    return True

def _fits_in_outer(inner_verts, outer_verts, eps=1e-9):
    """Check if all vertices of inner hexagon are inside (or on boundary of) outer hexagon."""
    for i in range(6):
        if not _point_in_polygon(inner_verts[i][0], inner_verts[i][1], outer_verts):
            return False
    # Also check that no edge of inner crosses outside — for convex polygons, 
    # if all vertices are inside, the polygon is inside.
    return True

def construct():
    start = time.time()
    DEADLINE = 115.0  # safety margin before 120s
    
    # Strategy: pack N=11 unit hexagons inside a regular hexagon.
    # We'll use a hexagonal lattice pattern with small rotations to improve packing.
    
    # Known good configurations:
    # A central hexagon surrounded by 6 in a ring, plus 4 more = 11 total.
    # This is a "flower" pattern: 1 center + 6 ring + 4 extra.
    
    # Let's try to find a tight configuration using optimization.
    # Variables: positions (x,y) and angles for each inner hexagon.
    # Objective: minimize the side length of the containing hexagon.
    
    # We'll use a multi-start approach with differential evolution.
    
    def _compute_required_side(positions, angles, outer_angle=0.0):
        """Compute the minimum side length of a regular hexagon (centered at origin, with given rotation)
        that contains all inner hexagons. Returns side length, or inf if not feasible."""
        # For a given outer hexagon (center at origin, side s, rotation outer_angle),
        # we need all inner hexagons to be inside.
        # We binary search on s.
        
        def _is_inside(s):
            outer_verts = _hex_vertices(0.0, 0.0, s, outer_angle)
            for i in range(N):
                inner_verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                if not _fits_in_outer(inner_verts, outer_verts):
                    return False
            return True
        
        # Lower bound: the outer hex must contain the center, so s >= 1 (for a single hex)
        # Upper bound: large enough
        lo = 2.0
        hi = 20.0
        
        # First check if hi works
        if not _is_inside(hi):
            return hi * 2.0  # signal infeasible
        
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if _is_inside(mid):
                hi = mid
            else:
                lo = mid
        
        return hi
    
    def _objective(params):
        """Params: 3*N = [x0,y0,a0, x1,y1,a1, ...]"""
        positions = params[0::3].reshape(-1, 2)
        angles = params[1::3]
        
        # Check for overlaps between inner hexagons
        for i in range(N):
            v1 = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, N):
                v2 = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                if _polygons_overlap(v1, v2):
                    return 1e6  # penalize overlap
        
        # Compute required outer side length
        s = _compute_required_side(positions, angles)
        if s > 100:
            return 1e5
        return s
    
    # Initial guess: flower pattern
    # Center at (0,0), 6 around at distance 2*sqrt(3) ≈ 3.464 (for touching hexagons)
    # Actually for unit hexagons, center-to-center distance for touching is 2*sqrt(3)/1 * ... 
    # Let's think: regular hexagon side 1, width across flats = sqrt(3), width across corners = 2.
    # Two hexagons touch when center distance = sqrt(3) + sqrt(3) = 2*sqrt(3) ≈ 3.464? No.
    # The distance between centers of two touching hexagons (same orientation) is 2 * (sqrt(3)/2 * 2) = 2*sqrt(3)? 
    # Actually, the "radius" to a vertex is 1 (side length), to a flat is sqrt(3)/2.
    # For two hexagons with same orientation, they touch when center distance = 2 * (sqrt(3)/2) * ... 
    # Let me just use a reasonable initial guess and let optimization refine.
    
    # Flower pattern: 1 center + 6 at 60-degree intervals at radius r
    # For unit hexagons, r ≈ 2*sqrt(3)/1... let's use r = 3.0 and let optimization adjust
    r = 3.0
    initial_positions = []
    initial_angles = []
    # Center
    initial_positions.append([0.0, 0.0])
    initial_angles.append(0.0)
    # Ring of 6
    for k in range(6):
        theta = np.radians(60 * k)
        initial_positions.append([r * np.cos(theta), r * np.sin(theta)])
        initial_angles.append(0.0)
    # 4 more: place them in the gaps
    # Gaps are at 30, 90, 150, 210, 270, 330 degrees at slightly larger radius
    for k in range(4):
        theta = np.radians(30 + 120 * k)  # 30, 150, 270, 30
        # Actually let's put them at 90, 210, 330, and 270... 
        # Let's use 4 positions at 45-degree offsets
        pass
    
    # Simpler: just use 11 positions in a hexagonal lattice
    # Hex lattice: rows offset
    # For 11 hexagons, use a 3-4-3 pattern or similar
    
    # Let's use a simple grid-like initial and let DE optimize
    np.random.seed(42)
    
    # Try multiple starting points
    best_result = None
    best_score = 1e10
    
    # Starting point 1: flower
    p1 = np.zeros(3 * N)
    p1[0:2] = [0, 0]
    p1[2] = 0
    for k in range(6):
        theta = 2 * np.pi * k / 6
        idx = (k + 1) * 3
        p1[idx] = 3.0 * np.cos(theta)
        p1[idx + 1] = 3.0 * np.sin(theta)
        p1[idx + 2] = 0
    # 4 more at larger radius
    for k in range(4):
        theta = 2 * np.pi * (k + 0.5) / 6 + np.pi / 6
        idx = (k + 7) * 3
        p1[idx] = 5.0 * np.cos(theta)
        p1[idx + 1] = 5.0 * np.sin(theta)
        p1[idx + 2] = 0
    
    # Starting point 2: linear-ish
    p2 = np.zeros(3 * N)
    for k in range(N):
        p2[k * 3] = (k - 5) * 3.5
        p2[k * 3 + 1] = 0
        p2[k * 3 + 2] = 0
    
    # Starting point 3: random
    p3 = np.zeros(3 * N)
    p3[0::3] = np.random.uniform(-6, 6, N)
    p3[1::3] = np.random.uniform(-6, 6, N)
    p3[2::3] = np.random.uniform(0, 180, N)
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 180))   # angle
    
    for start_params in [p1, p2, p3]:
        if time.time() > DEADLINE:
            break
        
        # Use differential evolution with a fixed number of iterations
        # Limit population and iterations to stay within time budget
        try:
            result = differential_evolution(
                _objective,
                bounds,
                x0=start_params,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return a trivially valid (but large) configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions = best_result[0::3].reshape(-1, 2)
    angles = best_result[1::3]
    
    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(angles[i])] for i in range(N)]
    
    # Compute the actual required side length for the final configuration
    s = _compute_required_side(positions, angles)
    
    if s > 100 or not np.isfinite(s):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (0.0, 0.0), float(s), 0.0
