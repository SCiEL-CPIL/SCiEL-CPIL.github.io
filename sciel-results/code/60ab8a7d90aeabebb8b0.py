"""Optimize placement of 12 unit hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = np.radians(angle_deg)
    angles = a + np.linspace(0, 2*np.pi, 7)[:-1]
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(pt, center, side, angle_deg):
    """Check if a point is inside a regular hexagon (distance-based for flat-top)."""
    # Transform point to hexagon's local frame
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The hexagon in local coords: |lx| <= s*sqrt(3)/2 and |ly| <= s*sqrt(3)/2 and |lx + ly| <= s*sqrt(3) ... 
    # Actually, for a hexagon with vertices at angles 0,60,...,300 from center at distance s:
    # The constraint is that for each edge, the point must be on the correct side.
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 (for pointy-top) 
    # Let's use the standard: vertices at 0,60,120,180,240,300 degrees from center
    # The 6 edge constraints: for edge between vertex i and i+1, normal points inward
    # For a hexagon with circumradius = side (for regular hexagon, circumradius = side):
    # Actually for regular hexagon, circumradius = side length.
    # Vertices at angle 0, 60, ...: (s*cos(k*60), s*sin(k*60))
    # The inradius (apothem) = s * cos(30) = s * sqrt(3)/2
    # The 6 supporting lines: for k=0..5, the line perpendicular to direction (30+k*60) at distance s*sqrt(3)/2
    # So constraint: dot(p, n_k) <= s*sqrt(3)/2 for all k, where n_k = (cos(30+k*60), sin(30+k*60))
    # But this is for a hexagon centered at origin with specific orientation.
    # Let me just check: for our hex with vertices at angle a + k*60, the edges are perpendicular to directions a + 30 + k*60
    # The apothem direction for edge k is a + 30 + k*60
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        n_angle = a + np.radians(30 + 60 * k)
        nx, ny = np.cos(n_angle), np.sin(n_angle)
        d = lx * nx + ly * ny
        if d > apothem:
            return False
    return True

def hexagon_containment_violation(inner_verts, outer_center, outer_side, outer_angle):
    """Return max violation (positive if any inner vertex is outside outer hex)."""
    max_viol = 0.0
    apothem = outer_side * np.sqrt(3) / 2
    a = np.radians(outer_angle)
    for v in inner_verts:
        dx, dy = v[0] - outer_center[0], v[1] - outer_center[1]
        ca, sa = np.cos(-a), np.sin(-a)
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        for k in range(6):
            n_angle = a + np.radians(30 + 60 * k)
            nx, ny = np.cos(n_angle), np.sin(n_angle)
            d = lx * nx + ly * ny
            viol = d - apothem
            if viol > max_viol:
                max_viol = viol
    return max_viol

def hexagon_pair_violation(verts1, verts2):
    """Return max overlap violation between two hexagons (positive if overlapping)."""
    # Check if any vertex of one is inside the other, and vice versa
    # For simplicity, check vertex-in-polygon
    max_viol = 0.0
    # This is expensive; for optimization we use a simpler metric
    # Actually, let's just check distances between centers as a proxy during optimization
    return 0.0  # placeholder

def objective(params, fixed_outer=True):
    """
    params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12] for inner hexagons
    Returns: outer_side + penalty for violations
    """
    n_params = 3 * N
    inner_x = params[0:n_params:3]
    inner_y = params[1:n_params:3]
    inner_a = params[2:n_params:3]
    
    # Compute all inner hexagon vertices
    all_inner_verts = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        all_inner_verts.append(verts)
    
    # Find bounding hexagon: try to fit all points in a hexagon
    # The minimal enclosing hexagon (with fixed orientation) can be found by
    # trying different orientations and computing the minimal side length
    
    # For a hexagon with orientation theta, the side length needed is:
    # max over all points of max over 6 edge normals of (dot(p - c, n_k) + apothem_offset)
    # Actually, for a hexagon centered at c with orientation theta and side s:
    # For each point p and each edge direction k: dot(p-c, n_k) <= s*sqrt(3)/2
    # So s >= max_k max_p dot(p-c, n_k) / (sqrt(3)/2)
    # And also s >= max_k max_p dot(c-p, n_k) / (sqrt(3)/2) [for the opposite side]
    # Wait, the constraint is |dot(p-c, n_k)| <= s*sqrt(3)/2 for all k? No.
    # For a regular hexagon, the 6 constraints are dot(p-c, n_k) <= apothem for k=0..5
    # where n_k are the outward normals. This automatically handles both sides since
    # n_{k+3} = -n_k.
    # So s >= max_k max_p dot(p-c, n_k) / (sqrt(3)/2)
    
    # We also need to choose the center c. The optimal c for a given orientation
    # minimizes the max constraint.
    
    # For efficiency, let's fix the outer orientation to 0 and optimize center and side
    # Or better: let's compute the minimal enclosing hexagon for a set of points
    
    # Try several orientations for the outer hex
    best_side = np.inf
    for outer_angle in [0, 10, 20, 30, 40, 50, 60]:
        a = np.radians(outer_angle)
        normals = []
        for k in range(6):
            n_angle = a + np.radians(30 + 60 * k)
            normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
        
        # For each point, compute the 6 dot products
        all_dots = []
        for i in range(N):
            p = np.array([inner_x[i], inner_y[i]])
            dots = np.array([np.dot(p, n) for n in normals])
            all_dots.append(dots)
        
        # The center c should minimize max_k (max_p dot(p, n_k) - dot(c, n_k)) 
        # subject to dot(c, n_k) - min_p dot(p, n_k) also being bounded
        # Actually: we need dot(p_i - c, n_k) <= apothem for all i,k
        # => dot(c, n_k) >= dot(p_i, n_k) - apothem for all i
        # => dot(c, n_k) <= dot(p_i, n_k) + apothem for all i
        # So for each k: max_i(dot(p_i, n_k)) - apothem <= dot(c, n_k) <= min_i(dot(p_i, n_k)) + apothem
        # For feasibility: max_i(dot(p_i, n_k)) - min_i(dot(p_i, n_k)) <= 2*apothem
        # The minimal apothem (and hence side) for this orientation is:
        # apothem >= max_k (max_i(dot(p_i, n_k)) - min_i(dot(p_i, n_k))) / 2
        
        apothem_needed = 0.0
        for k in range(6):
            vals = np.array([np.dot(np.array([inner_x[i], inner_y[i]]), normals[k]) for i in range(N)])
            spread = np.max(vals) - np.min(vals)
            if spread / 2 > apothem_needed:
                apothem_needed = spread / 2
        
        side_needed = apothem_needed / (np.sqrt(3) / 2)
        if side_needed < best_side:
            best_side = side_needed
    
    # Also need to check inner hexagon overlaps - use a simple distance-based penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((inner_x[i]-inner_x[j])**2 + (inner_y[i]-inner_y[j])**2)
            # Two unit hexagons: minimum center distance for no overlap is sqrt(3) (edge sharing)
            # But with rotation, the minimum can be less. Conservative: 2*inradius = sqrt(3)
            min_dist = np.sqrt(3)  # inradius * 2
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    return best_side + 10.0 * overlap_penalty

def construct():
    start_time = time.time()
    
    # Generate initial placements on hexagonal lattice
    # Hex lattice: points at (i*dx + j*dx/2, j*dy) where dx=sqrt(3), dy=1.5 for unit hexagons
    # Actually for unit hexagons (side=1), the lattice spacing is:
    # Horizontal: sqrt(3), Vertical: 1.5 (for pointy-top) or similar
    
    # Let's create several initial configurations
    def lattice_init(rows, cols, offset=0):
        """Generate hex lattice positions."""
        dx = np.sqrt(3)
        dy = 1.5
        positions = []
        for r in range(rows):
            for c in range(cols):
                x = c * dx + (r % 2) * dx / 2
                y = r * dy
                positions.append([x, y, 0.0])
        return positions
    
    # Try different arrangements
    initializations = []
    
    # 2x6 grid
    for r in range(2):
        for c in range(6):
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2
            y = r * 1.5
            initializations.append([x, y, 0.0])
    
    # 3x4 grid
    init34 = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2
            y = r * 1.5
            init34.append([x, y, 0.0])
    
    # 4x3 grid
    init43 = []
    for r in range(4):
        for c in range(3):
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2
            y = r * 1.5
            init43.append([x, y, 0.0])
    
    # Compact hexagonal cluster: center + ring of 6 + partial second ring
    # Center at (0,0), first ring at distance sqrt(3) in 6 directions
    hex_cluster = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_cluster.append([x, y, 0.0])
    # Add 5 more from second ring (distance 3)
    for k in range(5):
        angle = np.radians(60 * k + 30)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        hex_cluster.append([x, y, 0.0])
    
    # Linear arrangement
    linear = [[k * np.sqrt(3), 0.0, 0.0] for k in range(12)]
    
    all_inits = [initializations, init34, init43, hex_cluster, linear]
    
    best_result = None
    best_score = np.inf
    
    for init in all_inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        params0 = np.array(init).flatten()
        
        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return the linear arrangement
        inner = [[k * np.sqrt(3), 0.0, 0.0] for k in range(12)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    # Extract best solution
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        a = best_result[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    # Compute the actual outer hexagon parameters
    # Find minimal enclosing hexagon
    best_outer_side = np.inf
    best_outer_center = (0, 0)
    best_outer_angle = 0
    
    inner_x = np.array([p[0] for p in inner])
    inner_y = np.array([p[1] for p in inner])
    
    for outer_angle in np.arange(0, 60, 5):
        a = np.radians(outer_angle)
        normals = []
        for k in range(6):
            n_angle = a + np.radians(30 + 60 * k)
            normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
        
        apothem_needed = 0.0
        for k in range(6):
            vals = inner_x * normals[k][0] + inner_y * normals[k][1]
            spread = np.max(vals) - np.min(vals)
            if spread / 2 > apothem_needed:
                apothem_needed = spread / 2
        
        side_needed = apothem_needed / (np.sqrt(3) / 2)
        if side_needed < best_outer_side:
            best_outer_side = side_needed
            # Center is midpoint of the extent
            # For the dominant direction
            best_outer_angle = outer_angle
    
    # Compute center as centroid of the bounding region
    cx = (np.max(inner_x) + np.min(inner_x)) / 2
    cy = (np.max(inner_y) + np.min(inner_y)) / 2
    
    # Refine: do a few more optimization steps
    if time.time() - start_time < DEADLINE - 5:
        try:
            result2 = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_result = result2.x
                inner = []
                for i in range(N):
                    x = best_result[3*i]
                    y = best_result[3*i+1]
                    a = best_result[3*i+2]
                    inner.append([float(x), float(y), float(a)])
                inner_x = np.array([p[0] for p in inner])
                inner_y = np.array([p[1] for p in inner])
                
                best_outer_side = np.inf
                for outer_angle in np.arange(0, 60, 5):
                    a = np.radians(outer_angle)
                    normals = []
                    for k in range(6):
                        n_angle = a + np.radians(30 + 60 * k)
                        normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
                    
                    apothem_needed = 0.0
                    for k in range(6):
                        vals = inner_x * normals[k][0] + inner_y * normals[k][1]
                        spread = np.max(vals) - np.min(vals)
                        if spread / 2 > apothem_needed:
                            apothem_needed = spread / 2
                    
                    side_needed = apothem_needed / (np.sqrt(3) / 2)
                    if side_needed < best_outer_side:
                        best_outer_side = side_needed
                        best_outer_angle = outer_angle
                
                cx = (np.max(inner_x) + np.min(inner_x)) / 2
                cy = (np.max(inner_y) + np.min(inner_y)) / 2
        except Exception:
            pass
    
    return inner, (float(cx), float(cy)), float(best_outer_side), float(best_outer_angle)
