"""
Puzzle-Solver approach for packing 11 unit regular hexagons into a minimal regular hexagon.

Strategy:
1. Generate candidate topological arrangements (clusters) of 11 hexagons on the hexagonal lattice.
   We focus on compact shapes that are likely to be near-optimal.
2. For each arrangement, the positions are fixed relative to each other (rigid body) but the
   whole cluster can be rotated and translated.
3. The objective is to minimize the side length of the outer regular hexagon that contains the
   union of the inner hexagons.
4. The outer hexagon can also be rotated. So we have two rotation angles:
   - alpha: rotation of the inner cluster
   - beta: rotation of the outer hexagon
5. For fixed alpha and beta, the minimal translation is determined by the support function.
   The minimal side length L is such that the outer hexagon (centered at some point, rotated by beta,
   side length L) contains all inner hexagons (rotated by alpha, at fixed relative positions).

   Actually, the outer hexagon's center is also a variable. But we can optimize over the center
   as well. However, since the outer hexagon is convex and the inner set is compact, the optimal
   center is the one that minimizes the maximum "distance" in the 6 directions of the hexagon.

   For a regular hexagon with side length L and rotation beta, the set of points (x, y) inside it
   satisfies:
     For each of the 6 directions theta_k = beta + k * 60 degrees, k=0..5:
       x * cos(theta_k) + y * sin(theta_k) <= L * cos(30 degrees) = L * sqrt(3)/2

   Wait, let's derive this carefully.

   A regular hexagon with side length L can be viewed as the intersection of 6 half-planes.
   The distance from the center to each side is the apothem, which is L * sqrt(3) / 2.
   The normals to the sides point in directions that are 60 degrees apart.

   If the hexagon is rotated by angle beta, the normals are at angles:
     beta + 0, beta + 60, beta + 120, beta + 180, beta + 240, beta + 300 degrees.

   So a point (x, y) is inside the hexagon centered at (cx, cy) with side length L and rotation beta
   if and only if:
     For all k in 0..5:
       (x - cx) * cos(beta + k * pi/3) + (y - cy) * sin(beta + k * pi/3) <= L * sqrt(3) / 2

   Let d_k = L * sqrt(3) / 2. Then we need:
     (x - cx) * cos(theta_k) + (y - cy) * sin(theta_k) <= d_k for all k.

   This is equivalent to:
     max_k [ (x - cx) * cos(theta_k) + (y - cy) * sin(theta_k) ] <= d_k

   For a fixed set of inner hexagon vertices (after rotation by alpha and translation by (tx, ty)),
   the maximum over all vertices and all k of [ (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) ]
   must be <= L * sqrt(3) / 2.

   We want to minimize L, which is equivalent to minimizing:
     max_{v in V, k in 0..5} [ (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) ]

   where V is the set of all vertices of all inner hexagons.

   For fixed alpha and beta, this is a convex optimization problem in (cx, cy, L).
   Specifically, we want to minimize L subject to:
     (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) <= L * sqrt(3) / 2 for all v, k.

   This can be rewritten as:
     L >= max_{v, k} [ (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) ] / (sqrt(3)/2)

   So for fixed cx, cy, the minimal L is:
     L(cx, cy) = max_{v, k} [ (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) ] / (sqrt(3)/2)

   And we want to minimize L(cx, cy) over cx, cy.

   This is a 2D convex optimization problem. The function L(cx, cy) is the maximum of affine functions,
   so it is convex. We can solve it using scipy.optimize.minimize with method='Nelder-Mead' or
   'Powell' (since it's not smooth), or we can use a more specialized method.

   However, since we have only 2 variables (cx, cy), we can use a simple grid search or a local
   optimization.

   But we also have alpha and beta to optimize over. So the total search space is 4D: (alpha, beta, cx, cy).

   Given the time limit, we need to be efficient.

   Alternative approach:
   For fixed alpha and beta, the optimal (cx, cy) can be found by solving a linear programming problem:
     Minimize L
     Subject to:
       (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) <= L * sqrt(3) / 2 for all v, k.

   This is a linear program in variables cx, cy, L. We can use scipy.optimize.linprog.

   But we have to do this for many (alpha, beta) pairs.

   Let's think about the ranges:
   - alpha: 0 to 60 degrees (by symmetry of the hexagon, we only need 60 degrees)
   - beta: 0 to 60 degrees (same reason)

   We can discretize alpha and beta, and for each pair, solve the LP to get the optimal L.

   Then we take the minimum L over all (alpha, beta) pairs.

   The number of vertices: each hexagon has 6 vertices, 11 hexagons have 66 vertices.
   The LP has 66 * 6 = 396 constraints, and 3 variables (cx, cy, L). This is small and fast.

   Let's implement this.

   Steps:
   1. Generate candidate clusters of 11 hexagons on the hexagonal lattice.
   2. For each cluster:
      a. Get the relative positions of the 11 hexagons (centers).
      b. For each hexagon center, generate the 6 vertices (for a unit hexagon with side length 1).
      c. The vertices are in a local coordinate system. When we rotate the cluster by alpha,
         all vertices are rotated by alpha.
      d. For each (alpha, beta) in a grid:
         - Rotate all vertices by alpha.
         - Set up the LP:
           Variables: cx, cy, L
           Minimize: L
           Constraints: For each vertex (vx, vy) and each k in 0..5:
             (vx - cx) * cos(beta + k*pi/3) + (vy - cy) * sin(beta + k*pi/3) <= L * sqrt(3)/2
           This can be rewritten as:
             -cos(theta_k) * cx - sin(theta_k) * cy - (sqrt(3)/2) * L <= -vx * cos(theta_k) - vy * sin(theta_k)
         - Solve the LP.
         - Record the optimal L.
      e. Keep the best L for this cluster.
   3. Return the best cluster and the corresponding (alpha, beta, cx, cy, L).

   However, the return interface requires:
     inner: list of [x, y, angle_degrees] for each hexagon
     center: (x, y) of the outer hexagon
     side: side length of the outer hexagon
     angle: rotation of the outer hexagon

   So we need to output the absolute positions of the inner hexagons (after rotation and translation).

   For a given cluster with relative centers (dx_i, dy_i), rotation alpha, and outer center (cx, cy),
   the absolute center of inner hexagon i is:
     (cx + dx_i * cos(alpha) - dy_i * sin(alpha), cy + dx_i * sin(alpha) + dy_i * cos(alpha))
   and its angle is alpha (in degrees).

   Wait, the inner hexagons can also be individually rotated. But in our cluster model, we assume
   they are all aligned with the lattice, so their relative rotation is 0. When we rotate the whole
   cluster by alpha, each hexagon is also rotated by alpha. So the angle for each inner hexagon
   is alpha (in degrees).

   But the problem says "each inner hexagon may translate and rotate". So we are allowed to rotate
   each hexagon independently. However, in our cluster model, we are fixing their relative rotations
   to be 0 (i.e., all aligned with the lattice). This is a restriction.

   To allow independent rotations, we would need to optimize over 11 additional angles, which is
   infeasible. So we stick with the assumption that all inner hexagons have the same rotation as
   the cluster. This is a reasonable heuristic.

   Let's implement the above.

   First, we need to generate candidate clusters. We can use a backtracking algorithm to generate
   all connected sets of 11 cells on the hexagonal lattice. But there are many such sets.

   Alternatively, we can start with a known good configuration and perturb it.

   For simplicity, let's generate a few standard shapes:
   - A 3x4 rectangle minus one corner (but on hex lattice, this is tricky).
   - A "flower" shape: 1 center + 6 neighbors + 4 more.
   - A line of 11.
   - A 2x6 rectangle (12 cells, remove one).

   Actually, let's use a BFS/DFS to generate all connected sets of 11 cells within a bounding box.

   But this might be too many. Let's limit the bounding box to, say, 5x5 or 6x6.

   Given time constraints, let's just use a few hand-picked shapes.

   Shape 1: A "bar" of 11 in a row.
   Shape 2: A "T" shape or cross.
   Shape 3: A compact 3x4 block.

   Let's define the hexagonal lattice coordinates. We'll use axial coordinates (q, r) where:
     x = q + r/2
     y = r * sqrt(3)/2

   The neighbors of (q, r) are:
     (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)

   Let's generate a few shapes:

   Shape 1: 11 in a row: (0,0), (1,0), (2,0), ..., (10,0)
   Shape 2: A 3x4 block: rows r=0,1,2,3 and columns q=0,1,2,3, but that's 16. Let's take a 3x3 block (9) plus 2 more.
   Shape 3: A "plus" shape: center (0,0), and its 6 neighbors, and 4 more.

   Let's just use Shape 1 and a compact shape.

   Actually, let's use a simple connected shape: a 2x6 rectangle (12 cells) minus one cell.

   For now, let's use two shapes:
   1. A line of 11.
   2. A "blob" of 11.

   We'll define them in axial coordinates.

   Let's code the LP solver part first, and test with a simple shape.

   Note: The LP might be unbounded or infeasible if the constraints are not consistent, but for
   reasonable shapes and L, it should be fine. We'll add a large bound on L to ensure feasibility.

   Let's implement.
"""

import numpy as np
from scipy.optimize import linprog
import time
import itertools
from collections import deque

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    # Generate candidate clusters
    clusters = generate_clusters()

    best_L = float('inf')
    best_cluster = None
    best_alpha = 0.0
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Grid for alpha and beta
    alpha_vals = np.linspace(0, np.pi/3, 13)  # 0 to 60 degrees, 13 points
    beta_vals = np.linspace(0, np.pi/3, 13)

    for cluster in clusters:
        if time.time() > deadline:
            break
        # Convert axial coordinates to Cartesian
        centers = axial_to_cartesian(cluster)
        # Generate all vertices
        vertices = generate_vertices(centers)

        for alpha in alpha_vals:
            if time.time() > deadline:
                break
            # Rotate vertices by alpha
            rot_vertices = rotate_points(vertices, alpha)

            for beta in beta_vals:
                if time.time() > deadline:
                    break
                # Solve LP for this (alpha, beta)
                L, cx, cy = solve_lp(rot_vertices, beta)
                if L < best_L:
                    best_L = L
                    best_cluster = cluster
                    best_alpha = alpha
                    best_beta = beta
                    best_cx = cx
                    best_cy = cy

    if best_cluster is None:
        # Fallback: return a simple line
        cluster = [(i, 0) for i in range(N)]
        best_alpha = 0.0
        best_beta = 0.0
        best_cx = 0.0
        best_cy = 0.0
        best_L = 40.0  # Large fallback

    # Construct the result
    centers = axial_to_cartesian(best_cluster)
    inner = []
    for (q, r) in best_cluster:
        x, y = axial_to_cartesian([(q, r)])[0]
        # Rotate and translate
        x_rot = x * np.cos(best_alpha) - y * np.sin(best_alpha)
        y_rot = x * np.sin(best_alpha) + y * np.cos(best_alpha)
        x_abs = best_cx + x_rot
        y_abs = best_cy + y_rot
        inner.append([x_abs, y_abs, np.degrees(best_alpha)])

    return inner, (best_cx, best_cy), best_L, np.degrees(best_beta)

def generate_clusters():
    """Generate a list of candidate clusters as lists of (q, r) axial coordinates."""
    clusters = []

    # Shape 1: Line of 11
    clusters.append([(i, 0) for i in range(N)])

    # Shape 2: A 2x6 block minus one (11 cells)
    # Let's make a 2x6 block: rows 0,1 and cols 0..5
    block = [(q, r) for r in [0, 1] for q in range(6)]
    # Remove one cell to make it 11
    # Remove (0, 0) for variety
    clusters.append(block[1:])  # Remove first element

    # Shape 3: A "plus" or cross shape
    # Center at (0,0), neighbors, and a few more
    # Let's define a specific shape
    plus = [(0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # 6 neighbors
            (2, 0), (0, 2), (-2, 2), (-2, 0)]  # 4 more
    clusters.append(plus)

    # Shape 4: A compact 3x3 block (9) plus 2 more
    block33 = [(q, r) for r in range(3) for q in range(3)]
    # Add 2 more adjacent cells
    block33.append((3, 1))
    block33.append((3, 2))
    clusters.append(block33)

    # Shape 5: A "staircase"
    stair = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]
    clusters.append(stair)

    return clusters

def axial_to_cartesian(axial_coords):
    """Convert axial coordinates (q, r) to Cartesian (x, y)."""
    result = []
    for q, r in axial_coords:
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        result.append((x, y))
    return result

def generate_vertices(centers):
    """Generate the 6 vertices for each unit hexagon centered at the given centers."""
    vertices = []
    # Vertices of a unit hexagon centered at origin, with one vertex pointing right
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees, with distance 1 from center
    # But wait, the side length is 1, so the distance from center to vertex is 1.
    # The vertices are:
    #   (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    base_vertices = [
        (1.0, 0.0),
        (0.5, np.sqrt(3)/2),
        (-0.5, np.sqrt(3)/2),
        (-1.0, 0.0),
        (-0.5, -np.sqrt(3)/2),
        (0.5, -np.sqrt(3)/2)
    ]

    for cx, cy in centers:
        for vx, vy in base_vertices:
            vertices.append((cx + vx, cy + vy))

    return vertices

def rotate_points(points, angle):
    """Rotate a list of (x, y) points by angle (in radians)."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = []
    for x, y in points:
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        rotated.append((x_rot, y_rot))
    return rotated

def solve_lp(vertices, beta):
    """
    Solve the LP to find the minimal L and the optimal center (cx, cy)
    for the given rotated vertices and outer hexagon rotation beta.
    """
    # Variables: cx, cy, L
    # Minimize: L
    # Constraints: For each vertex (vx, vy) and each k in 0..5:
    #   (vx - cx) * cos(theta_k) + (vy - cy) * sin(theta_k) <= L * sqrt(3)/2
    # where theta_k = beta + k * pi/3
    # Rewrite:
    #   -cos(theta_k) * cx - sin(theta_k) * cy - (sqrt(3)/2) * L <= -vx * cos(theta_k) - vy * sin(theta_k)

    n_vertices = len(vertices)
    n_constraints = n_vertices * 6

    # Objective: minimize L
    c = np.zeros(3)
    c[2] = 1.0  # Minimize L

    # Constraint matrix A and vector b
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)

    sqrt3_over_2 = np.sqrt(3) / 2.0

    for i, (vx, vy) in enumerate(vertices):
        for k in range(6):
            theta = beta + k * np.pi / 3.0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            row = i * 6 + k
            A[row, 0] = -cos_t
            A[row, 1] = -sin_t
            A[row, 2] = -sqrt3_over_2
            b[row] = -vx * cos_t - vy * sin_t

    # Bounds: cx, cy unbounded, L >= 0
    bounds = [(None, None), (None, None), (0, None)]

    # Solve LP
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            cx = res.x[0]
            cy = res.x[1]
            L = res.x[2]
            return L, cx, cy
        else:
            # If LP fails, return a large value
            return float('inf'), 0.0, 0.0
    except:
        return float('inf'), 0.0, 0.0

# EVOLVE-BLOCK-END
