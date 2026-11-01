"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        return np.array([[cx + side * np.cos(angle + i * np.pi / 3),
                          cy + side * np.sin(angle + i * np.pi / 3)] for i in range(6)])

    def min_penetration_depth(h1, h2):
        """
        h1, h2: each is (cx, cy, angle_deg)
        Return minimum penetration depth (positive if overlapping, <=0 if disjoint).
        Uses Minkowski difference: hex1 - hex2 should not contain origin.
        The Minkowski difference of two hexagons is a 12-gon (convex).
        We check if origin is inside using SAT.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx * dx + dy * dy

        # If centers are far apart, definitely disjoint
        # Max distance between centers of two unit hexagons when touching is 2*sqrt(3) ~ 3.464
        # But with rotation, the bounding circle radius is 2.0 (circumradius of unit hex)
        # Actually circumradius of unit hex is 1.0, so max center distance for touching is 2.0
        # Wait: circumradius of regular hex with side 1 is 1.0. So max distance is 2.0.
        # But the "width" in any direction is 2*sqrt(3)/2 * 1 = sqrt(3) ~ 1.732 for flat-to-flat
        # and 2.0 for point-to-point.
        # So if dist > 2.0, definitely disjoint.
        if dist_sq > 4.5:  # slightly larger for safety
            return -1.0

        # Get vertices of Minkowski difference: hex1 translated to origin - hex2 translated to origin
        # Actually, Minkowski diff: {p - q : p in hex1, q in hex2}
        # The vertices of Minkowski diff of two convex polygons are sums of vertices
        # For two hexagons, we need to compute the Minkowski difference properly.
        
        # Simpler: use SAT directly on the two hexagons to compute penetration depth.
        v1 = hex_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hex_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # SAT: for each edge normal, project both polygons
        # If any separating axis found, they are disjoint
        # Penetration depth = max over axes of (separation) ... actually
        # We want: for each axis, the overlap amount. If any axis has separation <= 0, disjoint.
        # The minimum penetration depth is the min over all axes of the "gap" where gap = max1 - min2
        # If gap <= 0 for any axis, they are disjoint (or touching).
        
        edges = []
        for i in range(6):
            edges.append(v1[i] - v1[(i + 1) % 6])
            edges.append(v2[i] - v2[(i + 1) % 6])
        
        min_gap = float('inf')
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap: if max1 < min2 or max2 < min1, they are separated on this axis
            # The "overlap" on this axis is: min(max1, max2) - max(min1, min2)
            # If this is <= 0, separated.
            overlap = min(max1, max2) - max(min1, min2)
            if overlap <= 0:
                return overlap  # Negative or zero means disjoint/touching
            min_gap = min(min_gap, overlap)
        
        return min_gap if min_gap < float('inf') else 0.0

    def objective(params):
        """
        params: 36 floats = 12 * (x, y, angle_deg)
        Returns outer side length if valid, else large penalty.
        """
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                pd = min_penetration_depth(inner[i], inner[j])
                if pd > 1e-6:
                    return 100.0 + pd  # Penalty
        
        # Compute minimum enclosing regular hexagon
        # The outer hexagon has center (cx, cy) and angle, side length s.
        # A point (px, py) is inside the outer hexagon if for all 6 edge normals n_k:
        #   n_k . ((px,py) - (cx,cy)) <= s * sqrt(3) / 2
        # where n_k is the outward normal of edge k.
        # To find minimum s enclosing all inner vertices:
        #   s = max over all vertices p, all normals n_k of n_k . (p - center) / (sqrt(3)/2)
        # But center and angle are also variables... 
        # Actually, for a given set of points, the minimum enclosing regular hexagon
        # can be found by trying all 6 orientations? No, the orientation is fixed by the hexagon type.
        # Wait, the outer hexagon is a regular hexagon, so its orientation can vary.
        # But the problem says "minimizing the outer side length", and the outer hexagon can rotate.
        # So we need to find min over center (cx,cy) and angle theta of the minimum s.
        
        # For a fixed angle theta, the minimum enclosing hexagon centered at (cx,cy) has:
        # s = max_p max_k n_k(theta) . (p - c) / (sqrt(3)/2)
        # To minimize s over c: for each k, n_k . (p - c) = n_k . p - n_k . c
        # max_p (n_k . p - n_k . c) = max_p(n_k . p) - n_k . c
        # So s(theta, c) = max_k [max_p(n_k(theta).p) - n_k(theta).c] / (sqrt(3)/2)
        # To minimize over c: this is a convex optimization in c.
        # By symmetry, the optimal c for a given theta is the "center" that minimizes max_k.
        # Actually, for a regular hexagon, the optimal center for a given orientation is
        # the point that minimizes the maximum over k of [max_p(n_k.p) - n_k.c].
        
        # Simpler approach: since we're doing this inside the objective for each evaluation,
        # and we have only 12 hexagons = 72 vertices, we can do a small grid search over angle
        # and then solve for center.
        
        # Actually, let's simplify: fix the outer hexagon orientation to match the inner ones
        # (angle = 0) and just find the minimum bounding hexagon. This is a valid heuristic.
        # Better: try a few angles.
        
        all_verts = np.vstack([hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        # all_verts: (72, 2)
        
        sqrt3 = np.sqrt(3)
        best_side = float('inf')
        
        # Try a few angles for the outer hexagon
        for outer_angle in [0.0, 30.0]:
            # For this angle, find optimal center
            # s(c) = max_k max_p (n_k . p - n_k . c) / (sqrt3/2)
            # Let M_k = max_p(n_k . p), then s(c) = max_k (M_k - n_k . c) / (sqrt3/2)
            # We want to minimize max_k (M_k - n_k . c)
            # This is a linear program: minimize t subject to M_k - n_k . c <= t for all k
            # Variables: c_x, c_y, t
            # We can solve this with a few iterations of a simple method, or use the fact
            # that for a hexagon, the optimal center is related to the "hexagonal" median.
            
            # For now, use the centroid of all vertices as a good center estimate
            c = np.mean(all_verts, axis=0)
            
            max_ext = 0.0
            for k in range(6):
                normal_angle = np.radians(outer_angle) + np.radians(30) + k * np.pi / 3
                normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                extents = np.dot(all_verts - c, normal)
                max_e = np.max(extents)
                if max_e > max_ext:
                    max_e = max_e
                    max_ext = max_e
            
            side = max_ext * 2.0 / sqrt3
            if side < best_side:
                best_side = side
        
        return best_side

    def initial_guess():
        """Honeycomb-like initial arrangement"""
        params = np.zeros(3 * N)
        # Center hexagon
        params[0] = 0.0
        params[1] = 0.0
        params[2] = 0.0
        
        # Ring of 6
        for i in range(6):
            theta = i * np.pi / 3
            idx = 3 * (1 + i)
            params[idx] = 2.0 * np.cos(theta)
            params[idx + 1] = 2.0 * np.sin(theta)
            params[idx + 2] = 0.0
        
        # Second ring: 5 more
        for i in range(5):
            theta = i * 2 * np.pi / 5 + np.pi / 5
            idx = 3 * (7 + i)
            params[idx] = 4.0 * np.cos(theta)
            params[idx + 1] = 4.0 * np.sin(theta)
            params[idx + 2] = 0.0
        
        return params

    # Bounds for differential evolution
    # Positions: roughly in [-8, 8]
    # Angles: [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-8, 8))   # x
        bounds.append((-8, 8))   # y
        bounds.append((0, 360))  # angle

    init = initial_guess()

    best_result = None
    best_obj = float('inf')

    # Run differential evolution with time limit
    def timed_objective(x):
        if time.time() > deadline - 2.0:
            raise TimeoutError()
        return objective(x)

    try:
        result = differential_evolution(
            timed_objective,
            bounds,
            x0=init,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.8,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        if result.success or result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except (TimeoutError, Exception) as e:
        pass

    # If DE didn't find anything good, use local search from init
    if best_result is None or best_obj > 10.0:
        current = init.copy()
        current_obj = objective(current)
        best_local = current.copy()
        best_local_obj = current_obj
        
        T = 0.5
        step_size = 0.3
        for step in range(30000):
            if time.time() > deadline - 5.0:
                break
            
            new_params = current.copy()
            idx = random.randint(0, 3 * N - 1)
            new_params[idx] += random.gauss(0, step_size * (0.5 + 0.5 * T))
            
            try:
                new_obj = objective(new_params)
            except:
                continue
            
            if new_obj < current_obj or random.random() < np.exp(-(new_obj - current_obj) / T):
                current = new_params
                current_obj = new_obj
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local = current.copy()
            
            T *= 0.9995
            if T < 0.01:
                T = 0.01

        if best_local_obj < best_obj:
            best_obj = best_local_obj
            best_result = best_local

    # Extract final result
    if best_result is None:
        best_result = init

    inner = []
    for i in range(N):
        inner.append([float(best_result[3*i]), float(best_result[3*i+1]), float(best_result[3*i+2])])
    
    # Compute final side and center
    all_verts = np.vstack([hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
    center = tuple(np.mean(all_verts, axis=0).tolist())
    side = objective(best_result)
    angle = 0.0
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            pd = min_penetration_depth(inner[i], inner[j])
            if pd > 1e-4:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: larger valid box
        inner = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.732, 0.0],
                 [1.0, -1.732, 0.0], [-1.0, 1.732, 0.0], [-1.0, -1.732, 0.0],
                 [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.732, 0.0],
                 [3.0, -1.732, 0.0], [-3.0, 1.732, 0.0], [-3.0, -1.732, 0.0]]
        center = (0.0, 0.0)
        side = 6.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
