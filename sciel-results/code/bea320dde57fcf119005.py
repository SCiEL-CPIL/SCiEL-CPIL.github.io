import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def compute_outer_side(inner):
        """
        Compute the minimum side length S of a regular hexagon centered at origin
        with angle 0 (flat-top? no, pointy-top aligned with axes? 
        Actually, let's define the outer hexagon to have the same orientation as the lattice 
        or optimize it? 
        For simplicity and symmetry, assume outer hexagon is centered at (0,0) and 
        we can rotate it. However, due to symmetry of the hexagonal packing, 
        aligning the outer hexagon with the packing axes is optimal.
        
        The constraint is that all vertices of all inner hexagons must lie within 
        the outer hexagon.
        
        If the inner hexagons are rotated by 0 (same as outer), the problem simplifies.
        The apothem of the outer hexagon is S * sqrt(3)/2.
        The condition is that the maximum projection of any vertex of any inner hexagon
        onto the normals of the outer hexagon is <= S * sqrt(3)/2.
        
        Let's assume the outer hexagon has angle 0 (vertices at 0, 60, 120... or 30, 90...?
        Standard definition: 
        If angle=0, vertices are at 0, 60, 120, 180, 240, 300.
        Normals are at 30, 90, 150, 210, 270, 330.
        
        We will calculate the required S for a given set of inner hexagons.
        """
        # We assume inner hexagons have side 1.
        # We want to find min S such that all inner hexes are inside outer hex.
        # Outer hex center (0,0), angle 0.
        
        max_proj = 0.0
        # Normals of outer hex (angle 0): 30, 90, 150, 210, 270, 330 degrees
        normal_angles = [30, 90, 150, 210, 270, 330]
        
        for (ix, iy, iangle) in inner:
            # Generate vertices of this inner hexagon
            verts = hex_vertices(ix, iy, 1.0, iangle)
            for (vx, vy) in verts:
                for na_deg in normal_angles:
                    na = math.radians(na_deg)
                    nx, ny = math.cos(na), math.sin(na)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # S * apothem >= max_proj  => S * sqrt(3)/2 >= max_proj
        S = max_proj / (math.sqrt(3) / 2)
        return S

    def check_validity(inner, S):
        """
        Check if the configuration is valid:
        1. All inner hexagons are inside the outer hexagon of side S.
        2. All inner hexagons are disjoint.
        """
        # 1. Inside check
        # We already computed S based on max projection, so they are inside by definition 
        # of how S was derived, provided we use the same S.
        # But we need to ensure they don't overlap each other.
        
        # For disjointness, since all are regular hexagons of side 1,
        # if they have the same orientation, the minimum distance between centers 
        # must be at least 2 * apothem = sqrt(3).
        # If they have different orientations, it's more complex.
        # Given the symmetric nature of the optimal packing, they likely share the same orientation.
        # Let's enforce same orientation for the optimization variables to simplify, 
        # or check general disjointness.
        
        # General disjointness check is expensive. 
        # Let's assume for the optimization phase that they are all oriented at 0.
        # This is a strong assumption but likely true for the global minimum.
        # If we allow rotation, the search space is 3*N = 33 dimensions.
        # If we fix rotation to 0, it's 2*N = 22 dimensions.
        
        # Let's try fixing rotation to 0 first for speed and reliability.
        # If the optimal solution requires rotation, we might miss it, 
        # but for hexagonal packing in a hexagon, aligned is usually optimal.
        
        # Check pairwise distances
        n = len(inner)
        min_dist_sq = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        # Required distance is 2 * apothem = sqrt(3)
        # So dist >= sqrt(3) => dist_sq >= 3
        if min_dist_sq < 3.0 - 1e-9:
            return False
            
        return True

    def objective(params):
        """
        Objective function for scipy.optimize.
        params: array of length 2*N, [x1, y1, x2, y2, ..., xN, yN]
        We assume all hexagons are rotated by 0 degrees.
        """
        if time.time() > deadline:
            return 1e9
            
        inner = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner.append((x, y, 0.0))
        
        S = compute_outer_side(inner)
        
        if not check_validity(inner, S):
            # Penalty for overlap
            return S + 100.0
        else:
            return S

    # Initial configuration: Hexagonal lattice
    # Center + 6 neighbors + 4 from second ring?
    # Total 11.
    # Spacing d = sqrt(3) is the minimum. We will optimize positions.
    # Let's start with a perfect hexagonal lattice with spacing d = sqrt(3).
    
    init_params = np.zeros(2*N)
    
    # Generate hexagonal lattice points
    # Using axial coordinates (q, r)
    # (0,0) -> (0,0)
    # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Next ring: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    
    # We need 11 points.
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points. We need 4 more from Ring 2.
    
    points = []
    # Ring 0
    points.append((0,0))
    # Ring 1
    for q, r in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
        points.append((q,r))
        
    # Ring 2 (select 4 symmetrically, e.g., aligned with axes)
    # Axial to Cartesian (pointy top):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Wait, standard hex grid to pixel:
    # For pointy top:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size=1.
    
    ring2_candidates = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    # Pick 4 that are symmetric, e.g., (2,0), (0,2), (-2,2), (-2,0) ? No, let's pick (2,0), (1,1), (-1,2), (-2,1) ?
    # Let's just pick the first 4 in a symmetric pattern: (2,0), (0,2), (-2,2), (-2,0) is not symmetric.
    # (2,0), (1,1), (-1,2), (-2,1) forms a diamond?
    # Let's just pick 4 arbitrary from ring 2, optimization will fix it.
    # Actually, for symmetry, maybe (2,0), (0,2), (-2,2), (-2,0) is bad.
    # How about (2,0), (1,-1) is ring 1.
    # Let's take (2,0), (2,-2), (0,-2), (-2,0) ?
    # Let's just take (2,0), (1,1), (-1,2), (-2,1).
    selected_ring2 = [(2,0), (1,1), (-1,2), (-2,1)]
    points.extend(selected_ring2)
    
    # Ensure we have exactly N points
    while len(points) < N:
        # Add more from ring 2 if needed
        for p in ring2_candidates:
            if p not in points:
                points.append(p)
                if len(points) == N:
                    break
    points = points[:N]
    
    # Convert axial to cartesian
    for i, (q, r) in enumerate(points):
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        init_params[2*i] = x
        init_params[2*i+1] = y

    # Optimization
    try:
        # Use BFGS or Nelder-Mead
        # Nelder-Mead is robust for small dimensions without gradients
        # BFGS is faster if smooth
        # Let's try BFGS first, then Nelder-Mead if it fails or gets stuck
        
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_S:
            best_params = res.x
            best_S = res.fun
            best_inner = [(best_params[2*i], best_params[2*i+1], 0.0) for i in range(N)]
            best_center = (0.0, 0.0)
            best_angle = 0.0
            
    except Exception as e:
        pass

    # If the simple lattice optimization didn't find a good solution, 
    # try a few random restarts with Nelder-Mead
    for restart in range(5):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart)
        # Random perturbation of the lattice
        p = init_params.copy()
        for i in range(len(p)):
            p[i] += rng.gauss(0, 0.5)
            
        try:
            res = minimize(objective, p, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_S:
                best_params = res.x
                best_S = res.fun
                best_inner = [(best_params[2*i], best_params[2*i+1], 0.0) for i in range(N)]
                best_center = (0.0, 0.0)
                best_angle = 0.0
        except:
            continue

    if best_inner is None:
        # Fallback to initial lattice
        best_inner = [(init_params[2*i], init_params[2*i+1], 0.0) for i in range(N)]
        best_S = compute_outer_side(best_inner)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
