import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Vertices for angle 30 (flat-top)
    unit_verts_30 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])

    # For general angle, we might need to rotate. 
    # However, regular hexagons have 60-degree rotational symmetry.
    # So angle 0 is equivalent to 60, 120, etc.
    # Angle 30 is equivalent to 90, 150, etc.
    # Thus, we can restrict inner hexagon angles to {0, 30} without loss of generality 
    # if we allow continuous position optimization. 
    # Actually, rotating a hexagon by any amount is valid. 
    # But due to symmetry, the "shape" relative to the container is determined by angle mod 60.
    # Let's allow continuous angles in [0, 60) to cover all unique orientations.
    
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # Use unit verts for 0, rotate them
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        positions: (N, 2) array
        angles: (N,) array of angles in degrees
        """
        # The condition for a point p to be inside a regular hexagon centered at 0 
        # with side R and rotation theta is:
        # max_i ( n_i . p ) <= R * sqrt(3) / 2
        # where n_i are the outward normals of the hexagon edges.
        
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            # Get vertices of inner hexagon i
            verts = get_hex_verts(cx, cy, 1.0, ang)
            
            # Project all vertices onto the normals
            # proj = verts @ normals.T  -> (6, 6)
            # We need the max projection for each normal direction? 
            # No, for a convex polygon to be inside another convex polygon defined by half-planes,
            # every vertex of the inner polygon must satisfy the inequalities.
            # Inequality: n_j . p <= R * sqrt(3) / 2 for all j and all vertices p.
            # So R >= max( max_vertex( n_j . p ) ) * 2 / sqrt(3) for each j?
            # Actually, R must satisfy n_j . p <= R * sqrt(3) / 2 for ALL j and ALL p.
            # So R * sqrt(3) / 2 >= max( { n_j . p | j in 0..5, p in vertices } )
            
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(positions, angles):
        """
        Check if inner hexagons are disjoint.
        Returns True if disjoint, False if any overlap.
        """
        # Use Shapely for robust intersection check
        # Create polygons
        polys = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            # Ensure CCW order? Shapely handles it, but let's be safe.
            # unit_verts_0 is CCW. Rotation preserves orientation.
            polys.append(Polygon(verts))
        
        # Check pairwise
        for i in range(N):
            for j in range(i+1, N):
                if polys[i].intersects(polys[j]):
                    return False
        return True

    def objective(params):
        """
        Objective function for optimization.
        params: array of length 3N (x, y, angle for each hex)
        Returns: outer_side if valid, else large penalty.
        """
        positions = np.array(params[0:2*N]).reshape(N, 2)
        angles = np.array(params[2*N:3*N])
        
        # Check disjointness
        if not check_disjoint(positions, angles):
            return 1e6
        
        # Calculate minimal outer side
        side = calc_min_outer_side(positions, angles, 0.0)
        
        return side

    # Strategy:
    # 1. Use Differential Evolution to find a good configuration globally.
    #    Bounds:
    #    x, y: [-5, 5] (generous, hexagons are size 1, 12 of them need ~ 4-5 radius)
    #    angle: [0, 60]
    
    bounds = []
    for i in range(N):
        bounds.append((-5, 5)) # x
        bounds.append((-5, 5)) # y
        bounds.append((0, 60)) # angle
    
    # DE is stochastic, so we run it with a time limit or population limit.
    # Since DE can be slow, we run it for a fixed number of iterations or until time runs out.
    # scipy's DE doesn't have a direct time limit, but we can monitor.
    # Alternatively, we can run DE with a modest population and maxiter, then polish with Nelder-Mead.
    
    best_result = None
    best_score = 1e6
    
    # Run DE
    try:
        # Use a callback to monitor time? Or just run a bounded number of iterations.
        # 150s limit. DE with pop=50, maxiter=100 might take ~10-20s depending on N.
        # N=12, 36 variables.
        
        # Let's try a moderate DE run first.
        res_de = differential_evolution(
            objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=0.01, 
            polish=False # We will polish manually
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_result = res_de.x
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't find a good solution or took too long, try a few random starts with Nelder-Mead
    # Or just polish the DE result.
    
    if best_result is not None:
        # Polish with Nelder-Mead
        try:
            res_nm = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
        except Exception as e:
            print(f"NM failed: {e}")
    else:
        # Fallback: Try a few random starts with NM
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 10: break
            x0 = np.random.uniform(-3, 3, 2*N)
            a0 = np.random.uniform(0, 60, N)
            x0_full = np.concatenate([x0, a0])
            
            try:
                res_nm = minimize(
                    objective, 
                    x0_full, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_result = res_nm.x
            except:
                pass

    # Final Check and Formatting
    if best_result is None or best_score > 1e5:
        # Absolute fallback: Honeycomb
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        # Calculate side for this fallback
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        side = calc_min_outer_side(pos, ang, 0.0)
        return inner, (0.0, 0.0), side, 0.0

    # Extract results
    positions = np.array(best_result[0:2*N]).reshape(N, 2)
    angles = np.array(best_result[2*N:3*N])
    
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    # Final validation to ensure no overlap and correct side
    # If there's a tiny overlap due to numerical error, we might need to adjust.
    # But the objective function penalizes overlap heavily, so it should be clean.
    
    final_side = calc_min_outer_side(positions, angles, 0.0)
    
    # One last check: if the calculated side is smaller than what's needed due to floating point 
    # or if overlap exists, we might return an invalid solution. 
    # Let's do a quick disjoint check on the final result.
    if not check_disjoint(positions, angles):
        # If overlap, we have a problem. Try to expand? No, we minimize.
        # This case should be rare if DE/NM worked.
        # Fallback to honeycomb
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        final_side = calc_min_outer_side(pos, ang, 0.0)
        return inner, (0.0, 0.0), final_side, 0.0

    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
