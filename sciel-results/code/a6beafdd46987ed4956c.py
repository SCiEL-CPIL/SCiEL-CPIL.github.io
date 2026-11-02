"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Unit regular hexagon with side length 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # For a hexagon centered at origin with "flat top" or "pointy top"?
    # Let's use standard orientation: vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2).
    # This is a "pointy right" hexagon.
    # The distance from center to vertex is 1.
    # The distance from center to edge (apothem) is sqrt(3)/2.
    
    # To pack N hexagons of side 1 into a larger hexagon of side L.
    # If we align them all with the same orientation, the problem becomes packing points in a hexagonal lattice-like structure.
    # Actually, if all inner hexagons have the same orientation, the non-overlap constraint is that the distance between centers must be at least 2 * apothem? No.
    # The Minkowski sum of two identical hexagons of side 1 is a hexagon of side 2.
    # So, centers must be at least distance 2 * apothem apart?
    # Let's check. The width of a unit hexagon (distance between parallel sides) is 2 * apothem = sqrt(3).
    # If we place two such hexagons side-by-side, their centers must be separated by sqrt(3) in the direction perpendicular to the sides.
    # Generally, the centers of two non-overlapping unit hexagons (same orientation) must lie outside the Minkowski sum of one hexagon and the reflection of the other.
    # Since they are convex and centrally symmetric, the Minkowski sum is a hexagon of side 2 (scaled by 2).
    # The constraint is: center_i - center_j must NOT be in the interior of the Minkowski sum.
    # Equivalently, center_i must be outside the Minkowski sum centered at center_j.
    # This is a set of linear inequalities for each pair.
    
    # However, solving LP for N=11 with O(N^2) constraints is feasible.
    # But the outer hexagon side length L is the variable we want to minimize.
    # The outer hexagon is also a regular hexagon.
    # Let the outer hexagon have center C_out and side length L.
    # The inner hexagons must be contained within the outer hexagon.
    # Containment constraint: For each inner hexagon center c_i, the hexagon c_i + H (where H is unit hexagon) must be inside C_out + L*H.
    # This is equivalent to c_i being inside C_out + (L-1)*H.
    # So, we need to place N points c_i inside a hexagon of side (L-1), such that the pairwise non-overlap constraints are satisfied.
    # And we want to minimize L.
    # This is a non-linear problem because L scales the outer hexagon.
    # However, if we fix the orientation of the outer hexagon (e.g., 0 degrees), and the inner hexagons (e.g., 0 degrees),
    # the constraints are linear in c_i and L?
    # The containment constraint: c_i must be in the hexagon defined by C_out + (L-1)*H.
    # The hexagon H is defined by 6 linear inequalities.
    # If H is fixed orientation, the inequalities for the outer hexagon of side S=L-1 are:
    # n_k . (c_i - C_out) <= S * apothem(H) for k=0..5?
    # Actually, the support function of a hexagon of side s in direction n is s * (1 + cos(theta))/?
    # Let's use the explicit vertices.
    # Unit hexagon H (side 1) vertices: v_k = (cos(k*pi/3), sin(k*pi/3)) for k=0..5.
    # The hexagon is the convex hull of these.
    # The inequalities for a point p to be inside s*H (centered at 0) are:
    # For each edge, the dot product with the normal is bounded.
    # The normals to the edges of the unit hexagon are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Let's call these normals n_k.
    # The distance from origin to edge is apothem = sqrt(3)/2.
    # So, for a hexagon of side s, the distance is s * sqrt(3)/2.
    # Constraint: n_k . p <= s * sqrt(3)/2 for all k.
    
    # So, for inner center c_i to be inside outer hexagon of side L (centered at C_out):
    # n_k . (c_i - C_out) <= (L-1) * sqrt(3)/2  (assuming inner hexagon side 1 is contained in outer side L, so effective radius is L-1? No.)
    # Wait. If we have a hexagon of side 1 inside a hexagon of side L.
    # The "buffer" is L-1?
    # The Minkowski difference (L*H) - (1*H) is (L-1)*H.
    # So yes, c_i must be in (L-1)*H translated by C_out.
    
    # So the problem is:
    # Minimize L
    # Subject to:
    # 1. n_k . (c_i - C_out) <= (L-1) * A  for all i, k. (A = sqrt(3)/2)
    # 2. Non-overlap: c_i - c_j not in interior of 2*H.
    #    This means: n_k . (c_i - c_j) <= 2 * A  for all i!=j, k.
    #    Wait, the Minkowski sum of two unit hexagons is a hexagon of side 2.
    #    The condition for non-overlap is that the difference vector d = c_i - c_j is NOT in the interior of 2*H.
    #    Since 2*H is convex and contains the origin, d is outside 2*H if and only if there exists some normal n_k such that n_k . d > 2*A?
    #    No. d is in 2*H if and only if n_k . d <= 2*A for ALL k.
    #    So d is NOT in 2*H if there EXISTS k such that n_k . d > 2*A.
    #    This is a disjunctive constraint, not linear.
    #    However, we can enforce non-overlap by ensuring the distance between centers is large enough.
    #    The minimum distance between centers of two non-overlapping unit hexagons (same orientation) is 2 * apothem = sqrt(3) if they are touching side-to-side.
    #    But if they are rotated relative to each other, it's different. Here they are same orientation.
    #    Actually, the non-overlap constraint for convex polygons P and Q is that they don't intersect.
    #    For two identical hexagons, the centers must be at least distance d_min apart.
    #    What is d_min? It is the minimum distance between two points in the Minkowski sum 2*H? No.
    #    The centers must be separated by at least the width of the hexagon in some direction?
    #    Actually, the condition is simply that the convex hulls don't overlap.
    #    This is equivalent to: for all k, n_k . (c_i - c_j) <= 2*A is NOT sufficient to guarantee overlap.
    #    If n_k . (c_i - c_j) <= 2*A for all k, then c_i - c_j is in 2*H, which means the hexagons overlap (or touch).
    #    So we need to avoid this.
    #    This is a non-convex constraint.
    
    # Alternative approach:
    # Since N is small (11), we can try a grid search or local search.
    # Or, we can fix the positions of the inner hexagons in a hexagonal lattice pattern and optimize the outer size.
    # A known dense packing of hexagons is the hexagonal lattice.
    # In a hexagonal lattice of unit hexagons, the centers are at distance sqrt(3) apart in 3 directions.
    # Let's try to place N=11 hexagons in a hexagonal cluster.
    # A hexagonal number H_k = 3k(k-1)+1.
    # H_1 = 1
    # H_2 = 7
    # H_3 = 19
    # So 11 hexagons don't form a perfect hexagon.
    # We can take a cluster of 7 and add 4 more.
    
    # Let's generate coordinates for a hexagonal lattice.
    # Basis vectors: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 1.5).
    # Wait, distance between centers is sqrt(3).
    # Let's use axial coordinates (q, r).
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's try to find a compact arrangement of 11 points.
    # We can use a simple local search to minimize the maximum distance from the centroid, or directly minimize the enclosing hexagon size.
    
    # Let's define a function to compute the minimum side length L of a hexagon (orientation 0) enclosing a set of points.
    # The points are the centers of the inner hexagons.
    # The enclosing hexagon must contain the centers plus the "radius" of the inner hexagons.
    # As derived, if inner centers are at c_i, the outer hexagon of side L must contain c_i in (L-1)*H.
    # So we need to find the smallest L such that all c_i are in (L-1)*H (centered at C_out).
    # This is equivalent to finding the smallest hexagon (orientation 0) enclosing the points c_i, then adding 1 to the side length.
    # Wait, if the smallest hexagon enclosing the centers has side S, then L = S + 1.
    # How to find the smallest enclosing hexagon of fixed orientation?
    # The hexagon is defined by 6 half-planes.
    # n_k . (p - C) <= S * A.
    # We want to minimize S.
    # This is a linear programming problem!
    # Variables: C_x, C_y, S.
    # Constraints:
    # n_k . c_i - n_k . C <= S * A  =>  n_k . c_i - n_k . C - S * A <= 0
    # -n_k . c_i + n_k . C <= S * A  =>  -n_k . c_i + n_k . C - S * A <= 0
    # For all i, k.
    # Minimize S.
    
    # So, the algorithm is:
    # 1. Generate an initial set of N centers (e.g., hexagonal lattice).
    # 2. Use local search to move the centers to minimize S (the side length of the enclosing hexagon of the centers).
    # 3. The final answer is L = S + 1.
    # 4. The outer hexagon center is C, side is L, angle is 0.
    # 5. The inner hexagons are at c_i, angle 0.
    
    # Let's implement the LP for the enclosing hexagon.
    
    A = np.sqrt(3) / 2.0
    # Normals for the 6 edges of a unit hexagon (pointy right)
    # Vertices at 0, 60, 120...
    # Edges are perpendicular to normals at 30, 90, 150, 210, 270, 330.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def min_enclosing_hexagon_side(points):
        """
        Find the minimum side length S of a hexagon (orientation 0) enclosing the given points.
        Returns S and the center (Cx, Cy).
        """
        n_pts = len(points)
        if n_pts == 0:
            return 0.0, (0.0, 0.0)
        
        # Variables: Cx, Cy, S
        # Minimize S
        c = np.array([0.0, 0.0, 1.0])
        
        # Constraints:
        # For each point i and normal k:
        # n_kx * c_ix + n_ky * c_iy - n_kx * Cx - n_ky * Cy - A * S <= 0
        # -n_kx * c_ix - n_ky * c_iy + n_kx * Cx + n_ky * Cy - A * S <= 0
        
        A_mat = []
        b_vec = []
        
        for i in range(n_pts):
            px, py = points[i]
            for k in range(6):
                nx, ny = normals[k]
                # n . (p - C) - A*S <= 0
                # n_x * p_x + n_y * p_y - n_x * C_x - n_y * C_y - A * S <= 0
                A_mat.append([-nx, -ny, -A])
                b_vec.append(nx * px + ny * py)
                
                # -n . (p - C) - A*S <= 0
                # -n_x * p_x - n_y * p_y + n_x * C_x + n_y * C_y - A * S <= 0
                A_mat.append([nx, ny, -A])
                b_vec.append(-nx * px - ny * py)
        
        A_mat = np.array(A_mat)
        b_vec = np.array(b_vec)
        
        # S must be positive
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method='highs')
        
        if res.success:
            Cx, Cy, S = res.x
            return S, (Cx, Cy)
        else:
            # Fallback: large value
            return 1e10, (0.0, 0.0)

    # Generate initial points in a hexagonal lattice
    # We want N=11 points.
    # Let's generate a cluster.
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_xy(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate a list of axial coords for a hexagonal shape of radius 2 (which has 19 points)
    # and pick 11 of them, preferably the central ones.
    # Or just generate a grid and pick the closest to center.
    
    points_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Check if within hexagonal radius 2
            # Distance in axial coords: max(|q|, |r|, |q+r|) <= 2?
            # Actually, the hexagonal distance is (|q| + |r| + |q+r|) / 2
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            if dist <= 2.0:
                points_axial.append((q, r))
    
    # Convert to xy
    points_xy = [axial_to_xy(q, r) for q, r in points_axial]
    
    # We have 19 points. We need 11.
    # Sort by distance from origin and take the first 11.
    points_xy.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_points = [p for p in points_xy[:N]]
    
    # Local search to minimize the enclosing hexagon side
    # We will perturb the points and check if the side decreases.
    # Since the objective is non-smooth, we can use a simple random search or gradient-free method.
    # Given the small N, we can try moving each point slightly in random directions.
    
    current_points = np.array(initial_points)
    current_side, current_center = min_enclosing_hexagon_side(current_points)
    best_side = current_side
    best_points = current_points.copy()
    best_center = current_center
    
    step_size = 0.1
    iters = 0
    max_iters = 5000 # Adjust based on time
    
    while iters < max_iters:
        if time.time() > deadline:
            break
        
        # Pick a random point
        i = np.random.randint(N)
        # Pick a random direction
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_points = current_points.copy()
        new_points[i] += np.array([dx, dy])
        
        new_side, new_center = min_enclosing_hexagon_side(new_points)
        
        if new_side < best_side:
            best_side = new_side
            best_points = new_points.copy()
            best_center = new_center
            current_points = new_points
            current_side = new_side
            # Decrease step size slightly?
            # step_size *= 0.99
        else:
            # If no improvement, maybe reduce step size
            step_size *= 0.95
            if step_size < 1e-4:
                break
        
        iters += 1
    
    # Final result
    L = best_side + 1.0
    Cx, Cy = best_center
    
    # Construct inner hexagons
    inner = []
    for p in best_points:
        inner.append([p[0], p[1], 0.0])
    
    return inner, (Cx, Cy), L, 0.0
# EVOLVE-BLOCK-END
