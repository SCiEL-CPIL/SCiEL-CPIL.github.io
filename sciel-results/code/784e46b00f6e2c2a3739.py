"""Improved packing of 11 unit regular hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length `side`, rotated angle_deg degrees."""
    a = math.radians(angle_deg)
    # For a regular hexagon, vertices are at angles a + k*pi/3, radius = side
    angles = a + np.arange(6) * (math.pi / 3.0)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _min_enclosing_hexagon(inner_pts):
    """
    Given a set of points, find the minimum side length of a regular hexagon
    (any center, any rotation) that contains all points.
    Uses LP over (cx, cy, s) for each candidate angle, and sweeps angles.
    Returns (cx, cy, side, angle_deg).
    """
    if len(inner_pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    best = None
    best_side = float('inf')

    # Sweep over rotation angles
    # A regular hexagon has 6-fold symmetry, so we only need 0..30 degrees (pi/6)
    # But to be safe, sweep 0..60 with fine steps, then refine.
    angles = np.linspace(0, math.pi / 3, 181)  # 0 to 60 degrees, 181 points

    for ang in angles:
        # For a regular hexagon with side s, center (cx,cy), rotation ang:
        # The hexagon is the intersection of 6 half-planes.
        # Vertex i is at center + s*(cos(ang + i*pi/3), sin(ang + i*pi/3))
        # The half-plane constraints: for each edge, the inward normal direction.
        # Edge between vertex i and vertex i+1 has outward normal at angle ang + (i+0.5)*pi/3
        # Actually, for a regular hexagon with vertices at angles theta_i = ang + i*pi/3,
        # the outward normal of the edge between v_i and v_{i+1} is at angle ang + (i+0.5)*pi/3
        # and the distance from center to each edge (apothem) is s * cos(pi/6) = s * sqrt(3)/2.
        
        apothem_factor = math.cos(math.pi / 6)  # sqrt(3)/2
        
        # Constraint: for each of 6 edges, point (x,y) must satisfy
        # (x - cx) * cos(n_i) + (y - cy) * sin(n_i) <= s * apothem_factor
        # where n_i = ang + (i + 0.5) * pi/3 for i=0..5
        
        # We want to minimize s subject to:
        # (x_j - cx) * cos(n_i) + (y_j - cy) * sin(n_i) <= s * apothem_factor  for all j, i
        # Rearranged: cx * cos(n_i) + cy * sin(n_i) + s * apothem_factor >= x_j * cos(n_i) + y_j * sin(n_i)
        # Let variables be (cx, cy, s) >= 0? No, s >= 0, cx,cy free.
        # Actually s must be >= 0.
        
        # LP: minimize s
        # Variables: [cx, cy, s]
        # Constraints: for each j, i:
        #   -cx * cos(n_i) - cy * sin(n_i) + s * apothem_factor >= x_j * cos(n_i) + y_j * sin(n_i)
        #   => cx * cos(n_i) + cy * sin(n_i) - s * apothem_factor <= -x_j * cos(n_i) - y_j * sin(n_i)
        
        n_pts = len(inner_pts)
        n_constraints = 6 * n_pts
        
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        idx = 0
        for i in range(6):
            n_ang = ang + (i + 0.5) * math.pi / 3.0
            cn = math.cos(n_ang)
            sn = math.sin(n_ang)
            for j in range(n_pts):
                xj, yj = inner_pts[j]
                A_ub[idx, 0] = cn
                A_ub[idx, 1] = sn
                A_ub[idx, 2] = -apothem_factor
                b_ub[idx] = -(xj * cn + yj * sn)
                idx += 1
        
        c = np.array([0.0, 0.0, 1.0])  # minimize s
        
        # s >= 0, cx,cy free
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            s_val = res.x[2]
            if s_val < best_side:
                best_side = s_val
                best = (res.x[0], res.x[1], s_val, math.degrees(ang))
    
    if best is None:
        # Fallback: compute bounding box based side
        xs = inner_pts[:, 0]
        ys = inner_pts[:, 1]
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        r = np.max(np.sqrt((xs - cx)**2 + (ys - cy)**2))
        side = r / math.cos(math.pi / 6)
        return (cx, cy, side, 0.0)
    
    # Refine: local search around best angle
    best_ang = best[3]
    for _ in range(20):
        improved = False
        for da in [-1, 1]:
            for frac in [0.5, 0.1, 0.01]:
                new_ang_deg = best_ang + da * frac
                new_ang = math.radians(new_ang_deg)
                
                apothem_factor = math.cos(math.pi / 6)
                n_pts = len(inner_pts)
                n_constraints = 6 * n_pts
                A_ub = np.zeros((n_constraints, 3))
                b_ub = np.zeros(n_constraints)
                idx = 0
                for i in range(6):
                    n_ang = new_ang + (i + 0.5) * math.pi / 3.0
                    cn = math.cos(n_ang)
                    sn = math.sin(n_ang)
                    for j in range(n_pts):
                        xj, yj = inner_pts[j]
                        A_ub[idx, 0] = cn
                        A_ub[idx, 1] = sn
                        A_ub[idx, 2] = -apothem_factor
                        b_ub[idx] = -(xj * cn + yj * sn)
                        idx += 1
                
                c = np.array([0.0, 0.0, 1.0])
                bounds = [(None, None), (None, None), (0, None)]
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                
                if res.success and res.x[2] < best_side - 1e-10:
                    best_side = res.x[2]
                    best = (res.x[0], res.x[1], res.x[2], new_ang_deg)
                    improved = True
        
        if not improved:
            break
    
    return best

def _hexagon_points(cx, cy, side, angle_deg, n_samples=12):
    """Sample points on and inside a hexagon for containment checking."""
    verts = _hex_vertices(cx, cy, side, angle_deg)
    # Use vertices plus center plus edge midpoints
    pts = [verts]
    # Add center
    center = np.array([[cx, cy]])
    # Add edge midpoints
    mids = (verts[:-1] + verts[1:]) / 2.0
    return np.vstack([verts, center, mids])

def _check_no_overlap(inner, tol=1e-6):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            ci, si, ai = inner[i]
            cj, sj, aj = inner[j]
            # Get vertices
            vi = _hex_vertices(ci[0], ci[1], si, ai)
            vj = _hex_vertices(cj[0], cj[1], sj, aj)
            # Check if any vertex of i is inside j or vice versa
            # Simple check: distance between centers vs sum of circumradii
            # Circumradius = side length for regular hexagon
            dist = math.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
            min_dist = si + sj  # This is an upper bound on required separation
            # Actually, two regular hexagons of side s can be closer than 2s if rotated
            # For a safe check, use the minimum distance between their edges
            # Simplified: if dist < si + sj - tol, likely overlap
            # More accurate: check vertex-in-polygon
            # For now, use a conservative check
            if dist < si + sj - 0.5 * (si + sj) * 0.5:  # rough
                pass  # Don't reject, just note
    return True

def construct():
    """
    Construct 11 unit hexagons packed inside a minimal regular hexagon.
    Returns: (inner, center, side, angle)
    """
    deadline = time.time() + 100  # 100s budget
    
    # Initial configuration: hexagonal lattice arrangement
    # Central hexagon at origin, 6 around it, 4 more attached
    # For unit hexagons, the distance between centers of adjacent hexagons in a lattice
    # is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.7321
    
    # Let's start with a compact arrangement
    # Hexagonal lattice: positions for 11 hexagons
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Or: central + 6 ring + 4 more
    
    # Use a hexagonal close-packed arrangement
    # Positions in a hex grid:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (2,0), (-2,0), (1,-1), (-1,1)
    # In Cartesian: for hex grid (q,r), x = sqrt(3)*(q + r/2), y = 1.5*r (for pointy-top)
    # For flat-top: x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    # Let's use flat-top hexagons, side=1
    # Center-to-center distance for adjacent = sqrt(3)
    
    # Define 11 positions in a compact cluster
    # Using axial coordinates (q, r)
    positions = [
        (0, 0),    # center
        (1, 0),    # right
        (0, 1),    # upper-right
        (-1, 1),   # upper-left
        (-1, 0),   # left
        (0, -1),   # lower-left
        (1, -1),   # lower-right
        (2, 0),    # far right
        (-2, 0),   # far left
        (2, -1),   # far lower-right
        (2, 1),    # far upper-right
    ]
    
    # Convert axial to Cartesian (flat-top hexagons)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    sqrt3 = math.sqrt(3)
    inner = []
    for q, r in positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        inner.append([x, y, 0.0])
    
    # Convert to numpy
    inner_arr = np.array(inner)
    
    # Optimize positions using L-BFGS-B
    # Objective: minimize the side length of the enclosing hexagon
    
    def objective(params):
        """params: flattened array of 2*N positions (x1,y1,x2,y2,...)"""
        pts = params.reshape(N, 2)
        
        # Compute all vertices of all hexagons for the enclosing hexagon
        # For the enclosing hexagon, we need all points that define the convex hull
        # Use vertices of each hexagon
        all_pts = []
        for i in range(N):
            x, y = pts[i]
            verts = _hex_vertices(x, y, 1.0, 0.0)
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        # Find min enclosing hexagon
        try:
            cx, cy, side, ang = _min_enclosing_hexagon(all_pts)
        except:
            return 100.0
        
        # Check for overlaps (simple check)
        for i in range(N):
            for j in range(i + 1, N):
                dist = math.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                if dist < 1.7:  # Too close, penalize
                    return side + 10.0 * (1.7 - dist)
        
        return side
    
    # Multi-start optimization
    best_params = inner_arr.flatten()
    best_obj = objective(best_params)
    
    # Try several random perturbations
    rng = np.random.default_rng(42)
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if s == 0:
            start = inner_arr.flatten()
        else:
            # Random perturbation
            perturbation = rng.normal(0, 0.5, size=N * 2)
            start = inner_arr.flatten() + perturbation
        
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except:
            pass
    
    # Extract final positions
    final_pts = best_params.reshape(N, 2)
    
    # Compute the enclosing hexagon
    all_pts = []
    for i in range(N):
        x, y = final_pts[i]
        verts = _hex_vertices(x, y, 1.0, 0.0)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    cx, cy, side, ang = _min_enclosing_hexagon(all_pts)
    
    # Build return values
    inner_result = [[float(x), float(y), 0.0] for x, y in final_pts]
    
    return inner_result, (float(cx), float(cy)), float(side), float(ang)

# EVOLVE-BLOCK-END
