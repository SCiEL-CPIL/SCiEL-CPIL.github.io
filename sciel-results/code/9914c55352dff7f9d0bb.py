"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy:
    - Use a non-convex optimization approach to maximize minimum clearance
    - Each inner hexagon can translate and rotate
    - Use trust-region reflective algorithm with penalty for constraint violations
    - The outer hexagon is centered at origin with rotation 0 for simplicity
    
    A regular hexagon of side s has vertices at angles 0, 60, ..., 300 degrees
    at distance s from center (circumradius = s).
    
    For two convex polygons, the minimum distance can be computed via
    Minkowski sum and checking if origin is inside, or via support functions.
    
    For efficiency, we approximate the minimum distance between two rotated hexagons
    using their support functions in a set of directions.
    """
    n = N
    if n <= 0:
        return [], (0.0, 0.0), 0.0, 0.0
    
    # Precompute: a unit regular hexagon (side 1) has circumradius 1.
    # Vertices of unit hexagon (side=1, center at origin, flat-top orientation):
    # Actually, let's use pointy-top: vertices at angles 0, 60, 120, 180, 240, 300
    # with radius = side length = 1.
    # But "regular hexagon of side length 1" has circumradius = 1.
    # The inradius (apothem) = sqrt(3)/2.
    
    # For a hexagon with side s, circumradius = s, apothem = s*sqrt(3)/2.
    
    # We'll parameterize each inner hexagon by (x, y, theta) where theta is rotation in radians.
    # The outer hexagon has center (0,0), rotation 0, side length L (to be minimized).
    
    # Objective: minimize L such that all inner hexagons are inside outer hexagon
    # and pairwise non-overlapping.
    
    # For two convex polygons A and B, they are disjoint iff for some direction u,
    # support_A(u) + support_B(-u) < 0 ... actually they're disjoint iff the Minkowski
    # difference doesn't contain origin, which is equivalent to: exists u such that
    # h_A(u) + h_B(-u) < 0 where h is support function.
    
    # For a regular hexagon centered at c with rotation theta and side s:
    # h(u) = c·u + s * max_{k=0..5} (u · v_k(theta))
    # where v_k are the unit vectors to vertices.
    
    # Actually for a regular hexagon of circumradius R (= side length), the support function is:
    # h(u) = R * max_{k} (u · (cos(alpha_k), sin(alpha_k)))
    # where alpha_k = theta + k*pi/3
    
    # Let's use a grid of directions to approximate the minimum distance.
    # For two hexagons i, j with centers c_i, c_j, rotations th_i, th_j, side s_i=1, s_j=1:
    # Distance >= 0 iff for all directions u: 
    #   (c_i - c_j)·u + h_i(u) + h_j(-u) >= 0  ... no wait.
    
    # Two convex sets A, B are disjoint iff exists u: h_A(u) + h_B(-u) < 0
    # where A and B are centered at origin. For translated versions:
    # A + c_A and B + c_B are disjoint iff exists u: h_A(u) + h_B(-u) - (c_A - c_B)·u < 0
    
    # So the minimum "gap" is: min over u of [(c_A - c_B)·u - h_A(u) - h_B(-u)]
    # If this min > 0, they're disjoint with clearance = that min.
    
    # For containment in outer hexagon: inner hexagon i is inside outer hexagon iff
    # for all u: h_inner_i(u) + h_outer(-u) - 0 <= 0 ... 
    # Actually inner is inside outer iff for all u: h_inner(u) <= h_outer(u)
    # i.e., h_inner(u) - h_outer(u) <= 0 for all u.
    
    # Let's define the optimization variables:
    # x = [x_1, y_1, th_1, x_2, y_2, th_2, ..., x_n, y_n, th_n, L]
    # where L is the outer side length.
    
    # Objective: minimize L
    # Constraints:
    # 1. For each pair (i,j), i<j: min_u [(c_i-c_j)·u - h_i(u) - h_j(-u)] >= 0
    # 2. For each i: max_u [h_i(u) - h_outer(u)] <= 0
    
    # These are non-smooth constraints. We can use a penalty approach or
    # sample the constraints at a finite set of directions.
    
    # Use a finite set of directions for approximation.
    n_dirs = 36  # 36 directions, every 10 degrees
    angles = np.linspace(0, 2*np.pi, n_dirs, endpoint=False)
    dir_vecs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def hex_support(u, center, theta, side):
        """Support function of a regular hexagon (circumradius=side) at direction u."""
        # Vertices at angles theta + k*pi/3 for k=0..5
        # h(u) = center·u + side * max_k (u·(cos(theta+k*pi/3), sin(theta+k*pi/3)))
        alpha = theta + np.arange(6) * np.pi / 3
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        dots = u[0] * cos_alpha + u[1] * sin_alpha
        return center[0]*u[0] + center[1]*u[1] + side * np.max(dots)
    
    def outer_support(u, L):
        """Support function of outer hexagon (center at origin, rotation 0, side L)."""
        alpha = np.arange(6) * np.pi / 3
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        dots = u[0] * cos_alpha + u[1] * sin_alpha
        return L * np.max(dots)
    
    def compute_constraints(vars):
        """Compute constraint violations. Returns array of violations (should be <= 0)."""
        x = vars
        # Parse variables
        centers = []
        thetas = []
        for i in range(n):
            centers.append((x[3*i], x[3*i+1]))
            thetas.append(x[3*i+2])
        L = x[3*n]
        
        violations = []
        
        # Pairwise non-overlap: for each pair, check all directions
        for i in range(n):
            for j in range(i+1, n):
                ci = np.array(centers[i])
                cj = np.array(centers[j])
                diff = ci - cj
                for d in dir_vecs:
                    # h_i(d) for hexagon i centered at origin (we'll add center separately)
                    # Actually let's compute properly
                    hi_d = 1.0 * np.max(d[0]*np.cos(thetas[i] + np.arange(6)*np.pi/3) + d[1]*np.sin(thetas[i] + np.arange(6)*np.pi/3))
                    hj_negd = 1.0 * np.max((-d[0])*np.cos(thetas[j] + np.arange(6)*np.pi/3) + (-d[1])*np.sin(thetas[j] + np.arange(6)*np.pi/3))
                    # Constraint: diff·d - hi_d - hj_negd >= 0
                    val = diff[0]*d[0] + diff[1]*d[1] - hi_d - hj_negd
                    violations.append(-val)  # We want val >= 0, so -val <= 0
        
        # Containment: for each inner hexagon, check all directions
        for i in range(n):
            for d in dir_vecs:
                hi_d = 1.0 * np.max(d[0]*np.cos(thetas[i] + np.arange(6)*np.pi/3) + d[1]*np.sin(thetas[i] + np.arange(6)*np.pi/3))
                ho_negd = L * np.max((-d[0])*np.cos(np.arange(6)*np.pi/3) + (-d[1])*np.sin(np.arange(6)*np.pi/3))
                # Constraint: hi_d - ho_negd <= 0 ... wait, inner inside outer means h_inner(u) <= h_outer(u) for all u
                # So h_inner(u) - h_outer(u) <= 0
                val = hi_d - ho_negd
                violations.append(val)
        
        return np.array(violations)
    
    # Initial guess: place hexagons in a hexagonal lattice pattern
    # For N=11, a compact arrangement might be a central hexagon surrounded by 6,
    # plus 4 more. Let's try a hexagonal grid.
    
    # Hexagonal lattice spacing for unit hexagons: 
    # Adjacent hexagons in a honeycomb have center distance = sqrt(3) (for flat-top) 
    # or 2*apothem = sqrt(3) for pointy-top.
    # Actually for regular hexagons of side 1 packed in honeycomb, center-to-center = sqrt(3).
    
    # Let's use a simple initial placement: hexagonal grid
    # Positions for a hexagonal grid with spacing sqrt(3):
    # Row 0: x = k*sqrt(3), y = 0
    # Row 1: x = (k+0.5)*sqrt(3), y = 1.5
    # etc.
    
    # For N=11, let's do a 3-4-3 or 2-3-3-2 pattern or just a linear arrangement first.
    # Let's try a hexagonal cluster: center + 6 around + 4 more.
    
    sp = math.sqrt(3)  # spacing for honeycomb
    init_positions = []
    # Central
    init_positions.append((0.0, 0.0))
    # 6 around
    for k in range(6):
        angle = k * np.pi / 3
        init_positions.append((sp * math.cos(angle), sp * math.sin(angle)))
    # 4 more in the next ring
    for k in range(4):
        angle = (2*k + 1) * np.pi / 6 + np.pi/6  # offset angles
        # Actually let's just put them at distance 2*sp at various angles
        r = 2 * sp
        angle = k * np.pi / 2 + np.pi/4
        init_positions.append((r * math.cos(angle), r * math.sin(angle)))
    
    # Trim or pad to n
    while len(init_positions) < n:
        k = len(init_positions)
        angle = k * 2.399  # golden angle for spiral
        r = 1.5 * sp * math.sqrt(k)
        init_positions.append((r * math.cos(angle), r * math.sin(angle)))
    init_positions = init_positions[:n]
    
    x0 = np.zeros(3*n + 1)
    for i in range(n):
        x0[3*i] = init_positions[i][0]
        x0[3*i+1] = init_positions[i][1]
        x0[3*i+2] = 0.0  # initial rotation
    x0[3*n] = 5.0  # initial outer side length
    
    # Penalty function
    def penalty_objective(vars):
        L = vars[3*n]
        violations = compute_constraints(vars)
        # Penalty for constraint violations
        penalty = np.sum(np.maximum(0, violations)**2)
        return L + 100.0 * penalty
    
    # Try optimization
    best_x = x0.copy()
    best_L = x0[3*n]
    
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
    # with the penalty approach
    result = minimize(penalty_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
    
    if result.success or result.fun < penalty_objective(x0):
        best_x = result.x
        best_L = result.x[3*n]
    
    # Refine with a tighter check
    # Verify the solution
    final_L = best_x[3*n]
    
    # Build the inner list
    inner = []
    for i in range(n):
        x_i = best_x[3*i]
        y_i = best_x[3*i+1]
        th_i = best_x[3*i+2]
        angle_deg = np.degrees(th_i) % 360.0
        inner.append([float(x_i), float(y_i), float(angle_deg)])
    
    # Clamp L to a reasonable minimum
    if final_L < 1.0:
        final_L = 1.0
    
    return inner, (0.0, 0.0), float(final_L), 0.0
# EVOLVE-BLOCK-END
