"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = np.radians(angle_deg)
        angles = a + np.arange(6) * np.pi / 3.0
        return np.column_stack([cx + side * np.cos(angles),
                                cy + side * np.sin(angles)])
    
    def min_enclosing_hexagon(vertices_all):
        """Find minimum side-length regular hexagon (any rotation) containing all vertices.
        Returns (center_x, center_y, side_length, angle_deg)."""
        # For a regular hexagon to contain a set of points, we need to find
        # the smallest regular hexagon (over all rotations) that contains them.
        # A regular hexagon with center c, side s, rotation theta can be described
        # as intersection of 6 half-planes.
        # For a unit hexagon with rotation theta, the support function in direction phi is:
        # h(phi) = s * max(0, cos(phi - theta), cos(phi - theta + pi/3), ...)
        # Actually for a regular hexagon with circumradius R (= side for regular hex),
        # the width in direction phi is 2*R*max_{k} cos(phi - theta - k*pi/3) for the 6 normals.
        # The normals are at angles theta, theta+pi/3, theta+2pi/3, theta+pi, theta+4pi/3, theta+5pi/3
        # But actually the 6 face normals of a regular hexagon are at angles theta + k*pi/3 for k=0..5
        # Wait, let me reconsider. A regular hexagon has 6 faces. The outward normals of the faces
        # are at angles theta, theta+pi/3, ..., theta+5pi/3 where theta is the angle of the first face normal.
        # The distance from center to each face (apothem) is s * sqrt(3)/2 for a regular hexagon of side s.
        # So the hexagon is: {p : (p-c)·n_k <= apothem for all k=0..5}
        # where n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
        # apothem = s * sqrt(3)/2
        
        # To find min s such that all points are inside:
        # For a given theta, the minimum s is:
        # s >= max over all points p, all k: (p-c)·n_k / apothem_unit
        # where apothem_unit = sqrt(3)/2 (for unit side)
        # So s >= max_{p,k} (p-c)·n_k / (sqrt(3)/2) = (2/sqrt(3)) * max_{p,k} (p-c)·n_k
        # But we also need (p-c)·n_k >= -apothem, i.e., the constraint is |...| <= apothem
        # Actually: (p-c)·n_k <= apothem AND (p-c)·(-n_k) <= apothem
        # i.e., |(p-c)·n_k| <= apothem for all k... no, that's not right either.
        # The 6 constraints are: (p-c)·n_k <= apothem for k=0..5
        # where n_k are the 6 outward normals. Note n_{k+3} = -n_k.
        # So constraints: (p-c)·n_k <= apothem and (p-c)·(-n_k) <= apothem => |(p-c)·n_k| <= apothem
        # Wait no. The 6 normals are at angles theta, theta+pi/3, theta+2pi/3, theta+pi, theta+4pi/3, theta+5pi/3
        # n_3 = -n_0, n_4 = -n_1, n_5 = -n_2
        # So the constraints are: (p-c)·n_0 <= a, (p-c)·n_1 <= a, (p-c)·n_2 <= a,
        #   (p-c)·(-n_0) <= a, (p-c)·(-n_1) <= a, (p-c)·(-n_2) <= a
        # i.e., |(p-c)·n_k| <= a for k=0,1,2 where a = apothem = s*sqrt(3)/2
        # So s >= (2/sqrt(3)) * max_{p, k=0,1,2} |(p-c)·n_k|
        
        # For a given theta and center c, the required side length is:
        # s(theta, c) = (2/sqrt(3)) * max_{p in points, k=0,1,2} |(p-c)·n_k(theta)|
        # where n_k(theta) = (cos(theta+k*pi/3), sin(theta+k*pi/3))
        
        # To minimize s, we need to optimize over theta and c.
        # For a fixed theta, the optimal c is the center of the minimum enclosing 
        # "hexagonal" region, which is related to the Chebyshev center in the hexagonal norm.
        # This is a linear programming problem for fixed theta.
        # But we also need to optimize theta.
        
        # Simplification: for fixed theta, find c that minimizes max_{p,k} |(p-c)·n_k|
        # This is: minimize t subject to (p-c)·n_k <= t and -(p-c)·n_k <= t for all p,k
        # This is a linear program in variables c_x, c_y, t.
        
        # For efficiency, let's use a simpler approach: try several theta values and for each,
        # find the optimal center via LP or by iterating.
        
        points = vertices_all  # (M, 2) array
        
        best_s = np.inf
        best_result = None
        
        # Try a range of theta values
        for theta_deg in range(0, 30, 2):  # hexagon has 60-degree symmetry, so 0-60 suffices, use 0-30 for finer
            theta = np.radians(theta_deg)
            # 3 independent normals (the other 3 are negatives)
            normals = np.array([[np.cos(theta + k * np.pi / 3.0),
                                  np.sin(theta + k * np.pi / 3.0)] for k in range(3)])
            # normals: (3, 2)
            
            # For each point and each normal, compute the projection
            # We want to find c that minimizes max_{p,k} |dot(p-c, n_k)|
            # = minimize max_{p,k} |dot(p, n_k) - dot(c, n_k)|
            # Let v_{p,k} = dot(p, n_k), w_k = dot(c, n_k)
            # Minimize max_{p,k} |v_{p,k} - w_k|
            # This is a 1D-ish problem in 2D (c is 2D, but w is determined by c)
            
            # Use LP: minimize t s.t. v_{p,k} - w_k <= t, w_k - v_{p,k} <= t for all p,k
            # Variables: c_x, c_y, t (3 variables)
            # Constraints: for each (p,k): dot(p, n_k) - dot(c, n_k) <= t
            #   => dot(c, n_k) + t >= dot(p, n_k)  ... wait
            #   v_{p,k} - w_k <= t  =>  dot(p,n_k) - dot(c,n_k) <= t
            #   => -dot(c,n_k) - t <= -dot(p,n_k)  ... let me just set up the LP properly
            
            # Minimize t
            # subject to: dot(c, n_k) - t <= dot(p, n_k) for all p, k
            #              -dot(c, n_k) - t <= -dot(p, n_k) for all p, k
            #              (i.e., |dot(p-c, n_k)| <= t)
            
            # Using scipy linprog:
            # c = [cx, cy, t], minimize [0, 0, 1]
            # A_ub @ c <= b_ub
            # For each p, k: n_kx*cx + n_ky*cy - t <= dot(p, n_k)
            #              -n_kx*cx - n_ky*cy - t <= -dot(p, n_k)
            
            M = len(points)
            A_ub = np.zeros((2 * M * 3, 3))
            b_ub = np.zeros(2 * M * 3)
            
            idx = 0
            for p in points:
                for k in range(3):
                    nk = normals[k]
                    # Constraint 1: dot(c, nk) - t <= dot(p, nk)
                    A_ub[idx, 0] = nk[0]
                    A_ub[idx, 1] = nk[1]
                    A_ub[idx, 2] = -1.0
                    b_ub[idx] = np.dot(p, nk)
                    idx += 1
                    # Constraint 2: -dot(c, nk) - t <= -dot(p, nk)
                    A_ub[idx, 0] = -nk[0]
                    A_ub[idx, 1] = -nk[1]
                    A_ub[idx, 2] = -1.0
                    b_ub[idx] = -np.dot(p, nk)
                    idx += 1
            
            from scipy.optimize import linprog
            res = linprog(c=np.array([0, 0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                          method='highs', options={'presolve': True})
            
            if res.success:
                t_val = res.x[2]
                s_val = 2.0 / np.sqrt(3.0) * t_val
                if s_val < best_s:
                    best_s = s_val
                    best_result = (res.x[0], res.x[1], s_val, theta_deg)
        
        if best_result is None:
            # Fallback: use bounding box
            xs = points[:, 0]
            ys = points[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            # Rough estimate
            max_dist = np.max(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2))
            return (cx, cy), max_dist * 2.0 / np.sqrt(3.0) * 1.2, 0.0
        
        return (best_result[0], best_result[1]), best_result[2], best_result[3]
    
    def get_all_vertices(inner, side=1.0):
        """Get all vertices of all inner hexagons."""
        verts = []
        for (x, y, ang) in inner:
            v = hexagon_vertices(x, y, side, ang)
            verts.append(v)
        return np.vstack(verts)
    
    def check_disjoint(inner, side=1.0, tol=1e-4):
        """Check that no two inner hexagons overlap. Returns minimum distance between any pair (positive = disjoint)."""
        # For two regular hexagons, check if they overlap by checking vertex-in-polygon
        # and edge intersections. Simplification: check if any vertex of one is inside the other.
        # This is a sufficient but not necessary condition for non-overlap.
        # For a more robust check, we'd need full polygon intersection.
        # For now, use a simpler distance-based check.
        
        n = len(inner)
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                # Distance between centers
                dc = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # For two unit hexagons, if centers are far enough apart, they don't overlap.
                # The maximum distance from center to any point in a unit hexagon is 1 (circumradius).
                # So if dc > 2, they definitely don't overlap.
                # But they might overlap even if dc < 2.
                # A safer check: if dc > 2*sqrt(3)/2 * 2 ... hmm.
                # Actually, for two unit hexagons in any orientation, they are disjoint if
                # the distance between centers > 2 * (inradius) = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
                # No wait, that's not right either. The inradius (apothem) is sqrt(3)/2 ≈ 0.866.
                # Two hexagons are disjoint if the distance between centers > 2 * apothem = sqrt(3)
                # only if they are aligned the same way. In general, the minimum distance between
                # two disjoint unit hexagons (in any orientation) is at least the minimum distance
                # between two points on their boundaries.
                # 
                # For a robust check, let's just compute the actual minimum distance between the two polygons.
                # This is expensive but necessary for correctness.
                
                # Simplified: use the fact that for the optimizer, we'll use a penalty approach.
                # For the final check, we'll do a more careful test.
                if dc < 2.0:  # Only check if they might overlap
                    # Compute min distance between two hexagons
                    v1 = hexagon_vertices(x1, y1, side, a1)
                    v2 = hexagon_vertices(x2, y2, side, a2)
                    # Check if any vertex of v1 is inside v2 or vice versa
                    # Use point-in-polygon test
                    for pt in v1:
                        if point_in_polygon(pt, v2):
                            min_dist = 0.0
                            break
                    if min_dist == 0.0:
                        break
                    for pt in v2:
                        if point_in_polygon(pt, v1):
                            min_dist = 0.0
                            break
                    if min_dist == 0.0:
                        break
        return min_dist
    
    def point_in_polygon(pt, polygon):
        """Ray casting point in polygon test. polygon is (6,2) array."""
        x, y = pt
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def objective(params):
        """Objective: outer hexagon side length + penalty for overlaps."""
        # params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12] (36 params)
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append([x, y, a])
        
        # Get all vertices
        verts = get_all_vertices(inner)
        
        # Find min enclosing hexagon
        center, side_len, angle = min_enclosing_hexagon(verts)
        
        # Check for overlaps (penalty)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                dc = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dc < 2.0:
                    v1 = hexagon_vertices(x1, y1, 1.0, a1)
                    v2 = hexagon_vertices(x2, y2, 1.0, a2)
                    # Check vertex in polygon
                    for pt in v1:
                        if point_in_polygon(pt, v2):
                            overlap_penalty += 10.0 * (2.0 - dc)
                            break
                    for pt in v2:
                        if point_in_polygon(pt, v1):
                            overlap_penalty += 10.0 * (2.0 - dc)
                            break
        
        return side_len + overlap_penalty
    
    # Honeycomb initial configuration
    def honeycomb_init():
        """Create a honeycomb arrangement of 12 hexagons."""
        # Honeycomb: center + ring of 6 + partial ring of 5
        # Spacing: adjacent hexagons in honeycomb are sqrt(3) apart center-to-center
        # when sharing a full edge.
        s = np.sqrt(3)
        centers = []
        # Center
        centers.append((0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = k * np.pi / 3.0
            centers.append((s * np.cos(ang), s * np.sin(ang)))
        # Ring 2: 5 hexagons (to make 12 total)
        # Place them at the "corners" of the hexagonal arrangement
        for k in range(5):
            ang = (k + 0.5) * np.pi / 3.0  # offset by 30 degrees
            r = 2 * s
            centers.append((r * np.cos(ang), r * np.sin(ang)))
        
        inner = []
        for (cx, cy) in centers:
            inner.append([cx, cy, 0.0])
        return inner
    
    # Multi-start Nelder-Mead
    best_score = np.inf
    best_params = None
    
    np.random.seed(42)
    
    n_starts = 20
    for start in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if start == 0:
            inner0 = honeycomb_init()
        else:
            # Perturbed honeycomb
            inner0 = honeycomb_init()
            for i in range(N):
                inner0[i][0] += np.random.uniform(-0.5, 0.5)
                inner0[i][1] += np.random.uniform(-0.5, 0.5)
                inner0[i][2] = np.random.uniform(0, 60)
        
        params0 = np.array([v for hex_ in inner0 for v in hex_])
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: just return the honeycomb with a computed enclosing hexagon
        inner0 = honeycomb_init()
        verts = get_all_vertices(inner0)
        center, side_len, angle = min_enclosing_hexagon(verts)
        return inner0, center, side_len, angle
    
    # Extract final inner
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Compute final enclosing hexagon
    verts = get_all_vertices(inner)
    center, side_len, angle = min_enclosing_hexagon(verts)
    
    return inner, center, side_len, angle
# EVOLVE-BLOCK-END
