"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return vertices of a regular hexagon with center (cx, cy), side length 'side',
    and rotation angle in degrees.
    Vertices are at distance 'side' from center? No, for a regular hexagon with side s,
    the circumradius is s.
    So vertices are at distance s from center.
    Angle 0 means one vertex points to the right? Or flat top?
    Standard: vertex at angle 0 is (s, 0).
    """
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Check if point (px, py) is inside the hexagon.
    Uses the property that a regular hexagon is the intersection of 6 half-planes.
    The normals to the edges point outward.
    For a hexagon with circumradius R (which equals side length s),
    the distance from center to each edge (apothem) is s * sqrt(3)/2.
    
    The 6 edge normals are at angles: angle_deg + 30°, 90°, 150°, 210°, 270°, 330°
    Wait, if vertices are at 0, 60, 120... then edges are between them.
    The normal to the edge between vertex 0 and 1 points at 30°.
    So the 6 outward normals are at angles: angle_deg + 30° + k*60°.
    
    A point p is inside if for all k=0..5:
    dot(p - c, n_k) <= apothem
    where n_k is the unit normal at angle (angle_deg + 30 + k*60)°.
    """
    apothem = side * np.sqrt(3) / 2.0
    base_angle_rad = np.deg2rad(angle_deg + 30)
    rel_x = px - cx
    rel_y = py - cy
    
    # We can check all 6 constraints
    # n_k = (cos(base + k*pi/3), sin(base + k*pi/3))
    # constraint: rel_x * cos(theta) + rel_y * sin(theta) <= apothem
    
    # To be fast, we can vectorize or just loop. Since N is small, loop is fine.
    # But in the objective function, we call this many times, so speed matters.
    # Let's precompute cos/sin for the 6 directions if possible, but angle varies.
    
    for k in range(6):
        theta = base_angle_rad + k * np.pi / 3.0
        nx = np.cos(theta)
        ny = np.sin(theta)
        if rel_x * nx + rel_y * ny > apothem + 1e-9:
            return False
    return True

def _hexagon_distance_to_boundary(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """
    Calculate the minimum distance from the inner hexagon (center cx,cy, angle angle_deg)
    to the boundary of the outer hexagon (center outer_cx, outer_cy, side outer_side, angle outer_angle_deg).
    
    This is the minimum over all points in the inner hexagon of the distance to the outer hexagon boundary?
    No, we need to ensure the inner hexagon is INSIDE the outer hexagon.
    This is equivalent to: for every vertex of the inner hexagon, it must be inside the outer hexagon.
    And for every edge of the inner hexagon, it must not cross the outer hexagon boundary.
    Actually, since both are convex, it suffices to check that all vertices of the inner hexagon are inside the outer hexagon.
    
    So we can just check the 6 vertices of the inner hexagon against the outer hexagon.
    If any vertex is outside, the configuration is invalid.
    To make it a smooth objective, we can define the "violation" as the maximum distance
    of any inner vertex from the outer boundary (in the outward normal direction).
    If all are inside, the violation is negative (or 0).
    
    Let's define a smooth measure of how far the inner hexagon extends beyond the outer boundary.
    For each of the 6 vertices of the inner hexagon, compute the "distance" to the outer hexagon.
    If the vertex is inside, the distance is negative (depth).
    If outside, it's positive.
    We want the maximum of these 6 values to be <= 0.
    
    How to compute the signed distance from a point to a regular hexagon?
    It's the maximum over the 6 half-plane constraints:
    d_k = dot(p - outer_c, n_k_outer) - apothem_outer
    where n_k_outer are the outward normals of the outer hexagon.
    The point is inside if all d_k <= 0.
    The "distance" (signed) is max(0, max(d_k)).
    But for smoothness, we can use a soft-max or just the max of the d_k values.
    If max(d_k) <= 0, the point is inside.
    If max(d_k) > 0, the point is outside by max(d_k) (in the worst normal direction).
    
    So for the inner hexagon to be inside the outer, we need:
    For each of the 6 inner vertices v_i:
        max_k ( dot(v_i - outer_c, n_k_outer) - apothem_outer ) <= 0
    
    We can define the objective component for this inner hexagon as:
        max_i max_k ( dot(v_i - outer_c, n_k_outer) - apothem_outer )
    
    This is the maximum violation over all inner vertices and all outer half-planes.
    If this is <= 0, the inner hexagon is inside.
    
    However, we also need to minimize the outer side length.
    So the problem is:
    Minimize outer_side
    Subject to:
        1. Inner hexagons are disjoint.
        2. Inner hexagons are inside the outer hexagon.
    
    Since outer_side is a variable, and the constraints depend on it,
    we can fix the outer hexagon center and angle (by symmetry, we can fix them to 0,0 and 0°),
    and then minimize outer_side such that all constraints are satisfied.
    
    Actually, we can rephrase:
    For a given set of inner hexagon centers and angles,
    what is the minimum outer_side such that all inner hexagons fit inside?
    This is equivalent to finding the smallest regular hexagon (with fixed center and angle)
    that contains all inner hexagons.
    
    The smallest regular hexagon containing a set of points (or shapes) is determined by
    the maximum extent in the 6 normal directions.
    Specifically, for a regular hexagon with center C and angle A,
    the apothem required to contain a point P is:
        max_k ( dot(P - C, n_k) )
    where n_k are the 6 unit normals.
    The side length is then apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3).
    
    So for a set of points (the vertices of all inner hexagons),
    the minimum apothem is:
        max_{all vertices v} max_k ( dot(v - C, n_k) )
    And the side length is 2 * apothem / sqrt(3).
    
    This is a smooth function of the inner hexagon positions (since max is nonsmooth,
    but we can use a soft-max or just the max, which is piecewise linear).
    
    So the objective is:
        f(x) = 2 / sqrt(3) * max_{i,k} ( dot(v_{i,k} - C, n_j) )
    where v_{i,k} are the vertices of inner hexagon i, and n_j are the 6 outer normals.
    
    Wait, this gives the minimum outer hexagon that contains all vertices.
    But we also need the inner hexagons to be disjoint.
    
    So the full problem is:
    Minimize f(x) = 2 / sqrt(3) * max_{i,k,j} ( dot(v_{i,k} - C, n_j) )
    Subject to:
        For all i != l, inner hexagon i and l are disjoint.
    
    Disjointness of two convex polygons can be checked by the Separating Axis Theorem (SAT).
    Two convex polygons are disjoint if there exists a separating axis.
    For regular hexagons, the axes to check are the normals to the edges of both hexagons.
    So 6 + 6 = 12 axes.
    For each axis, project both hexagons onto the axis.
    If the projections are disjoint, the hexagons are disjoint.
    If all projections overlap, the hexagons intersect.
    
    To make this a smooth constraint, we can define the "overlap" on each axis.
    For a given axis n, the projection of hexagon i is [min_i, max_i].
    The projection of hexagon l is [min_l, max_l].
    The overlap is: min(max_i, max_l) - max(min_i, min_l).
    If overlap <= 0, they are disjoint on this axis.
    If overlap > 0, they overlap on this axis.
    
    The hexagons are disjoint if there exists at least one axis with overlap <= 0.
    This is an OR condition, which is hard to handle in optimization.
    
    Alternative: We can penalize the maximum overlap over all axes.
    If the maximum overlap is <= 0, they are disjoint.
    If the maximum overlap > 0, they intersect.
    So the constraint is: max_{axis} overlap(axis) <= 0.
    
    This is a smooth (piecewise linear) constraint.
    
    So the optimization problem is:
    Minimize f(x) = 2 / sqrt(3) * max_{i,k,j} ( dot(v_{i,k} - C, n_j) )
    Subject to:
        For all i < l, max_{axis} overlap_{i,l}(axis) <= 0
    
    Where x is the vector of all inner hexagon centers and angles.
    
    This is a non-smooth optimization problem (due to the max operations),
    but we can use scipy.optimize.minimize with a method that can handle non-smoothness,
    or we can smooth the max operations with a log-sum-exp.
    
    Given the time limit, let's try a simple approach:
    1. Fix the outer hexagon center to (0,0) and angle to 0°.
    2. Initialize the inner hexagons in a reasonable pattern.
    3. Use scipy.optimize.minimize to minimize the objective function,
       with the disjointness constraints as penalty terms.
    
    Let's implement this.
    
    """
    pass # This function is not used directly, the logic is in construct()

def _compute_objective_and_constraints(x, outer_cx=0.0, outer_cy=0.0, outer_angle_deg=0.0):
    """
    x: array of shape (N, 3), where x[i] = [cx, cy, angle_deg] for inner hexagon i.
    Returns:
        obj: the minimum outer side length required to contain all inner hexagons.
        constraints: array of shape (N*(N-1)//2,), where each element is the maximum overlap
                     between a pair of inner hexagons. We want these to be <= 0.
    """
    N = len(x)
    outer_side = None
    
    # Precompute outer hexagon normals
    outer_base_angle = np.deg2rad(outer_angle_deg + 30)
    outer_normals = np.array([
        [np.cos(outer_base_angle + k * np.pi / 3.0),
         np.sin(outer_base_angle + k * np.pi / 3.0)]
        for k in range(6)
    ])
    
    # Compute vertices of all inner hexagons
    all_vertices = []
    for i in range(N):
        cx, cy, angle_deg = x[i]
        verts = _hexagon_vertices(cx, cy, 1.0, angle_deg)
        all_vertices.append(verts)
    
    # Compute the maximum extent in each outer normal direction
    # For each vertex of each inner hexagon, compute dot(v - outer_c, n_j) for all j
    # The maximum of these values is the required apothem.
    max_extent = 0.0
    for i in range(N):
        verts = all_vertices[i]
        for v in verts:
            rel = v - np.array([outer_cx, outer_cy])
            dots = rel @ outer_normals.T  # shape (6,)
            max_extent = max(max_extent, np.max(dots))
    
    outer_side = 2.0 * max_extent / np.sqrt(3.0)
    
    # Compute disjointness constraints
    num_pairs = N * (N - 1) // 2
    constraints = np.zeros(num_pairs)
    pair_idx = 0
    for i in range(N):
        for l in range(i + 1, N):
            # Compute the maximum overlap between hexagon i and l
            # We need to check 12 axes: 6 from hex i, 6 from hex l
            verts_i = all_vertices[i]
            verts_l = all_vertices[l]
            
            # Get normals for hex i and l
            angle_i = x[i][2]
            angle_l = x[l][2]
            
            base_i = np.deg2rad(angle_i + 30)
            base_l = np.deg2rad(angle_l + 30)
            
            normals_i = np.array([
                [np.cos(base_i + k * np.pi / 3.0),
                 np.sin(base_i + k * np.pi / 3.0)]
                for k in range(6)
            ])
            normals_l = np.array([
                [np.cos(base_l + k * np.pi / 3.0),
                 np.sin(base_l + k * np.pi / 3.0)]
                for k in range(6)
            ])
            
            all_normals = np.vstack((normals_i, normals_l))
            
            max_overlap = -np.inf
            for n in all_normals:
                # Project hex i onto n
                proj_i = verts_i @ n
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                
                # Project hex l onto n
                proj_l = verts_l @ n
                min_l = np.min(proj_l)
                max_l = np.max(proj_l)
                
                # Overlap
                overlap = min(max_i, max_l) - max(min_i, min_l)
                max_overlap = max(max_overlap, overlap)
            
            constraints[pair_idx] = max_overlap
            pair_idx += 1
    
    return outer_side, constraints

def _objective(x):
    """
    Objective function: outer_side + penalty for constraint violations.
    """
    outer_side, constraints = _compute_objective_and_constraints(x)
    # Penalty for constraints > 0
    penalty = np.sum(np.maximum(0, constraints) ** 2)
    return outer_side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initialize inner hexagons
    # A reasonable initial guess: place them in a hexagonal grid pattern
    # For 12 hexagons, we can use a 2x6 or 3x4 grid, but hexagonal packing is better.
    # Let's start with a simple grid and let the optimizer fix it.
    x0 = np.zeros((N, 3))
    # Place them in a circle
    for i in range(N):
        angle = 2 * np.pi * i / N
        x0[i, 0] = 2.0 * np.cos(angle)
        x0[i, 1] = 2.0 * np.sin(angle)
        x0[i, 2] = 0.0  # Initial angle
    
    # Try multiple restarts with different initial guesses
    best_x = x0.copy()
    best_score = np.inf
    
    # Use Nelder-Mead or Powell, which can handle non-smooth functions
    # Nelder-Mead is derivative-free and robust
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the initial guess for restarts > 0
        if r == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + np.random.normal(0, 0.5, size=x0.shape)
        
        try:
            res = minimize(
                _objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < best_score:
                best_x = res.x
                best_score = res.fun
                
        except Exception:
            pass
    
    # Extract the best solution
    inner = best_x.tolist()
    center = (0.0, 0.0)
    
    # Compute the actual outer side length for the best solution
    outer_side, constraints = _compute_objective_and_constraints(best_x)
    
    # If constraints are violated, we need to scale up the outer hexagon
    # But the objective already includes the penalty, so outer_side should be reasonable.
    # To be safe, if there are violations, increase outer_side slightly.
    max_violation = np.max(np.maximum(0, constraints))
    if max_violation > 1e-6:
        outer_side += max_violation + 0.01
    
    return inner, center, outer_side, 0.0
# EVOLVE-BLOCK-END
