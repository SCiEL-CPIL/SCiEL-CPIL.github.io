import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices and normals
    # Vertices of a unit hexagon centered at (0,0) with angle 0
    # Angle 0 means a vertex is at (1, 0)? Or an edge is vertical?
    # Standard math: vertex at angle 0 is (1,0).
    angles = np.arange(6) * np.pi / 3
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Normals to the edges.
    # Edge between vertex i and i+1.
    # Normal angle is midpoint angle + 90 deg?
    # Vertex 0 at 0 deg, Vertex 1 at 60 deg. Midpoint 30 deg. Normal at 30+90=120?
    # Let's just use the vertices rotated by 30 degrees as normals?
    # For a regular polygon, the outward normal to the edge connecting $V_i$ and $V_{i+1}$
    # is in the direction of $V_{i+0.5}$.
    # So normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = angles + np.pi / 6
    unit_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def get_hex_verts(x, y, theta):
        """Return vertices of a hexagon at (x,y) with rotation theta."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_verts @ R.T + np.array([x, y])

    def get_hex_normals(theta):
        """Return normals of a hexagon with rotation theta."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_normals @ R.T

    def sat_disjoint(v1, n1, v2, n2):
        """
        Check if two hexagons are disjoint using SAT.
        v1, v2: (6, 2) arrays of vertices.
        n1, n2: (6, 2) arrays of normals.
        Returns True if disjoint.
        """
        all_normals = np.vstack((n1, n2))
        
        # Project v1 and v2 onto all normals
        # proj1: (12, 6) ? No.
        # v1 @ n.T gives (6, 6) matrix where [i, j] is dot(v1[i], n[j])
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # If for any axis, the intervals [min, max] do not overlap, they are disjoint
        # Overlap if max1 >= min2 and max2 >= min1
        # Disjoint if max1 < min2 or max2 < min1
        disjoint = (max1 < min2) | (max2 < min1)
        
        return np.any(disjoint)

    def compute_side(params):
        """
        Compute the minimum outer side length for the given inner hexagons.
        Assumes outer hexagon is centered at (0,0) and has normals at 30, 90, ...
        """
        max_extent = 0.0
        for i in range(N):
            x, y, theta = params[3*i], params[3*i+1], params[3*i+2]
            verts = get_hex_verts(x, y, theta)
            # Project onto the fixed outer normals
            # Outer normals are unit_normals (fixed orientation)
            proj = verts @ unit_normals.T
            extents = np.max(np.abs(proj), axis=0)
            local_max = np.max(extents)
            if local_max > max_extent:
                max_extent = local_max
        
        # Apothem a = max_extent
        # Side S = 2 * a / sqrt(3)
        return 2.0 * max_extent / np.sqrt(3)

    def objective(params):
        # Check disjointness
        for i in range(N):
            xi, yi, thi = params[3*i], params[3*i+1], params[3*i+2]
            vi = get_hex_verts(xi, yi, thi)
            ni = get_hex_normals(thi)
            for j in range(i+1, N):
                xj, yj, thj = params[3*j], params[3*j+1], params[3*j+2]
                vj = get_hex_verts(xj, yj, thj)
                nj = get_hex_normals(thj)
                
                if not sat_disjoint(vi, ni, vj, nj):
                    return 1e6
        
        return compute_side(params)

    def initial_flower():
        params = np.zeros(3*N)
        # Center
        params[0:3] = [0, 0, 0]
        # Ring 1: 6 hexagons
        # Use distance 2.1 to ensure strict disjointness if required
        for i in range(6):
            theta = i * np.pi / 3
            x = 2.1 * np.cos(theta)
            y = 2.1 * np.sin(theta)
            params[3 + 3*i : 6 + 3*i] = [x, y, 0]
        # Ring 2: 5 hexagons (approx)
        # Place them at distance 4.1, angles offset
        # We have 12 hexagons total.
        # Hex 0: Center
        # Hex 1-6: Ring 1 (indices 1-6)
        # Hex 7-11: Ring 2 (indices 7-11)
        for i in range(5):
            # Distribute 5 hexagons in the second ring
            # Angles offset to avoid overlap with Ring 1
            theta = i * 2 * np.pi / 5 + np.pi / 5
            # Distance 4.1 is safe from Ring 1 (dist 2.1) and Center (dist 0)
            x = 4.1 * np.cos(theta)
            y = 4.1 * np.sin(theta)
            idx = 7 + i
            params[3*idx : 3*idx+3] = [x, y, 0]
        return params

    def initial_grid():
        params = np.zeros(3*N)
        # 3 rows, 4 cols
        # Use spacing 2.1 x and 1.8 y to ensure disjointness
        for i in range(4):
            for j in range(3):
                idx = i * 3 + j
                x = i * 2.1
                y = j * 1.8
                params[3*idx : 3*idx+3] = [x, y, 0]
        return params

    def initial_random():
        params = np.random.rand(3*N) * 4 - 2
        params[2::3] = 0 # Fix angles initially to 0 for simplicity in random start?
        return params

    best_params = None
    best_score = np.inf

    initial_guesses = [
        initial_flower(),
        initial_grid(),
        initial_random()
    ]

    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        # Use Nelder-Mead
        res = minimize(objective, guess, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

        # Also try a few local restarts from the best found so far
        # If time permits, do 2 more restarts with small perturbations
        if best_params is not None and time.time() < deadline - 10:
             for _ in range(2):
                 perturbed = best_params + np.random.normal(0, 0.1, 3*N)
                 res2 = minimize(objective, perturbed, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
                 if res2.fun < best_score:
                     best_score = res2.fun
                     best_params = res2.x

    if best_params is None:
        # Fallback
        best_params = initial_flower()
        best_score = objective(best_params)

    # Extract inner hexagons
    inner = []
    for i in range(N):
        x, y, theta = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        # Convert theta radians to degrees
        inner.append([float(x), float(y), float(np.degrees(theta))])
    
    center = (0.0, 0.0)
    side = float(best_score)
    angle = 0.0
    
    return inner, center, side, angle
