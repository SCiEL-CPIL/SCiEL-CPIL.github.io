"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Hexagonal lattice basis vectors for unit hexagons (flat-top)
    # Distance between centers of touching flat-top hexagons is sqrt(3)
    # Basis vectors:
    # u1 = (0, sqrt(3))  (vertical neighbor)
    # u2 = (1.5, sqrt(3)/2) (diagonal neighbor)
    # Actually, standard axial coordinates (q, r) map to Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # This generates a lattice of centers for non-overlapping unit hexagons.
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y

    # Generate candidate clusters of size N on the hexagonal lattice
    # We restrict to "compact" shapes to minimize the bounding box.
    # We will generate all connected sets of size N within a small radius.
    # To keep it fast, we generate sets by expanding from the origin.
    
    def generate_clusters(n_size):
        clusters = set()
        
        # Start with the origin
        current = [(0, 0)]
        
        # We do a BFS/DFS expansion to find all connected shapes of size n_size
        # within a bounding box of roughly -n_size to n_size in axial coords.
        
        # This might be too slow if we do it naively for every run.
        # Instead, let's hardcode a few known good compact shapes for N=11.
        # Or, generate them once.
        
        # Let's generate all connected subsets of size 11 from the lattice
        # within a radius of 3 (axial distance).
        
        # Pre-compute neighbors for axial coords
        def neighbors(q, r):
            return [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
        
        # To avoid duplicates, we use a canonical representation (frozenset)
        # and ensure we only add neighbors that are "larger" in some ordering
        # or simply use a set of frozensets.
        
        # Given N=11 is small, the number of free hexagonal animals of size 11 is manageable (a few hundred/thousand).
        # We will generate them.
        
        all_shapes = set()
        
        # Start with shape of size 1
        current_shapes = {frozenset([(0,0)])}
        
        for size in range(1, n_size):
            next_shapes = set()
            for shape in current_shapes:
                # Get all cells in shape
                cells = list(shape)
                # Get all neighbors of all cells
                neighbor_set = set()
                for c in cells:
                    for nb in neighbors(c[0], c[1]):
                        if nb not in shape:
                            neighbor_set.add(nb)
                
                # Add each neighbor to form a new shape
                for nb in neighbor_set:
                    new_shape = frozenset(shape | {nb})
                    # Canonicalize: translate so that min x, then min y is at (0,0)?
                    # Or just use frozenset. Frozenset is hashable.
                    # But (0,0), (1,0) is different from (1,0), (2,0).
                    # We need to identify identical shapes up to translation.
                    # We can normalize by subtracting the min q, then min r.
                    
                    qs = [c[0] for c in new_shape]
                    rs = [c[1] for c in new_shape]
                    min_q = min(qs)
                    min_r = min(rs)
                    
                    normalized = frozenset((q - min_q, r - min_r) for q, r in new_shape)
                    next_shapes.add(normalized)
            
            current_shapes = next_shapes
            
        # current_shapes now contains all free hexagonal animals of size n_size
        return list(current_shapes)

    # Generate shapes
    shapes = generate_clusters(N)
    
    # For each shape, compute the minimal enclosing hexagon side length S
    # We assume the inner hexagons are all rotated 0 degrees (aligned with outer).
    # The outer hexagon is centered at (0,0) and rotated 0 degrees.
    # We need to find the minimum S such that all hexagons fit.
    # This is equivalent to finding the minimum S such that the union of the
    # unit hexagons (placed at lattice points) is contained within a regular
    # hexagon of side S.
    
    # Since the inner hexagons are convex and the outer is convex,
    # and we are scaling the outer hexagon, the condition is:
    # max_{p in union} support_function_outer(p) <= S
    # where support_function_outer(p) is the maximum distance from origin
    # in the directions normal to the sides of the outer hexagon, normalized.
    
    # For a regular hexagon of side S centered at origin (flat-top):
    # The boundaries are:
    # x <= S
    # -x <= S
    # sqrt(3)/2 * y + 1/2 * x <= S * sqrt(3)/2  => y + x/sqrt(3) <= S
    # -sqrt(3)/2 * y + 1/2 * x <= S * sqrt(3)/2 => -y + x/sqrt(3) <= S
    # sqrt(3)/2 * y - 1/2 * x <= S * sqrt(3)/2 => y - x/sqrt(3) <= S
    # -sqrt(3)/2 * y - 1/2 * x <= S * sqrt(3)/2 => -y - x/sqrt(3) <= S
    
    # Actually, the distance from origin to the side in direction n_i is S * sqrt(3)/2.
    # The support function of the outer hexagon in direction n_i is S * sqrt(3)/2.
    # So we need max_{p in union} dot(p, n_i) <= S * sqrt(3)/2 for all 6 normals.
    # Thus S >= max_i ( max_{p in union} dot(p, n_i) ) / (sqrt(3)/2)
    
    # The "union" consists of 11 unit hexagons.
    # The maximum of dot(p, n) over the union is the max over each hexagon.
    # For a single unit hexagon centered at c, the max of dot(p, n) is dot(c, n) + support_unit(n).
    # support_unit(n) for a unit flat-top hexagon is:
    # If n is along a normal to a side, support is sqrt(3)/2.
    # If n is along a vertex direction, support is 1.
    # The 6 normals of the outer hexagon are:
    # n1 = (1, 0) -> support_unit = 1 (vertex) ? No.
    # Let's check the geometry.
    # Flat-top hexagon vertices: (1,0), (0.5, sqrt3/2), ...
    # Normals to sides:
    # Side 1 (right, x=1): Normal (1,0). Max x on unit hex is 1. So support is 1.
    # Side 2 (top-right): Normal (1/2, sqrt3/2).
    #   Max of 0.5x + sqrt3/2 y on unit hex.
    #   At vertex (0.5, sqrt3/2): 0.5(0.5) + sqrt3/2(sqrt3/2) = 0.25 + 0.75 = 1.
    #   So support is 1.
    #   Wait, is the support of a unit hexagon in the direction of its side normal always 1?
    #   The apothem is sqrt3/2. The distance to the vertex is 1.
    #   The normal to the side points towards the midpoint of the side.
    #   The vertex is NOT in the direction of the side normal.
    #   Let's re-evaluate.
    #   Normal n = (1, 0). The point on the hexagon maximizing x is (1, 0). dot((1,0), (1,0)) = 1.
    #   Normal n = (1/2, sqrt3/2). The point maximizing this is the vertex (0.5, sqrt3/2)?
    #   dot((0.5, sqrt3/2), (0.5, sqrt3/2)) = 0.25 + 0.75 = 1.
    #   Yes, for a regular hexagon, the support function in the direction of any of the 6 vertex normals (which are also side normals for the dual?)
    #   Actually, the 6 normals to the sides of a flat-top hexagon are:
    #   (1, 0), (1/2, sqrt3/2), (-1/2, sqrt3/2), (-1, 0), (-1/2, -sqrt3/2), (1/2, -sqrt3/2).
    #   The vertices are at angles 0, 60, 120...
    #   The normal (1,0) points to the vertex (1,0). Support = 1.
    #   The normal (1/2, sqrt3/2) points to the vertex (0.5, sqrt3/2). Support = 1.
    #   So, support_unit(n_i) = 1 for all 6 outer normals.
    
    # Therefore, for a cluster of centers C, the required S is:
    # S = max_{i in 1..6} ( max_{c in C} (dot(c, n_i) + 1) ) / (sqrt(3)/2)
    # S = (2/sqrt3) * max_{i, c} (dot(c, n_i) + 1)
    
    # We want to minimize this S over all possible translations of the cluster.
    # Note: The cluster is defined by relative positions. We can translate the whole cluster.
    # Let the cluster positions be p_k. We can shift them by t.
    # New positions p_k + t.
    # S(t) = (2/sqrt3) * max_{i} ( max_k (dot(p_k + t, n_i) + 1) )
    # S(t) = (2/sqrt3) * max_{i} ( dot(t, n_i) + max_k dot(p_k, n_i) + 1 )
    
    # This is a convex optimization problem in t (minimizing the max of linear functions).
    # We can solve this using linear programming or simple ternary search if we restrict t?
    # Actually, since the domain is 2D, we can use scipy.optimize.minimize on this convex function.
    # Or, since it's the intersection of 6 half-planes in the "S-space", we can find the minimal S.
    
    # Let's use a simple gradient-free approach or linear programming.
    # Variables: t_x, t_y, S
    # Constraints:
    # For each i (0..5):
    #   dot(t, n_i) + max_k dot(p_k, n_i) + 1 <= S * sqrt(3)/2
    #   => dot(t, n_i) - S * sqrt(3)/2 <= - (max_k dot(p_k, n_i) + 1)
    
    # This is a Linear Program!
    # Minimize S
    # Subject to:
    #   n_i_x * t_x + n_i_y * t_y - (sqrt3/2) * S <= - (M_i + 1)  for i=0..5
    #   where M_i = max_k dot(p_k, n_i)
    
    # We can solve this LP for each shape.
    # Number of shapes is small.
    
    from scipy.optimize import linprog
    
    # Normals
    normals = []
    for k in range(6):
        angle = math.radians(60 * k)
        normals.append((math.cos(angle), math.sin(angle)))
    
    best_S = float('inf')
    best_shape = None
    best_t = (0, 0)
    
    # Precompute M_i for each shape? No, M_i depends on the shape.
    
    for shape in shapes:
        # Convert shape to Cartesian coordinates
        points = [axial_to_cartesian(q, r) for q, r in shape]
        
        # Calculate M_i = max_k dot(p_k, n_i)
        M = []
        for n in normals:
            max_dot = max(p[0]*n[0] + p[1]*n[1] for p in points)
            M.append(max_dot)
        
        # LP Setup
        # Variables: t_x, t_y, S
        # c = [0, 0, 1]
        
        # Constraints A_ub @ x <= b_ub
        # Rows for each of the 6 normals
        A_ub = []
        b_ub = []
        
        for i, n in enumerate(normals):
            # n_x * t_x + n_y * t_y - (sqrt3/2) * S <= - (M_i + 1)
            A_ub.append([n[0], n[1], -sqrt3/2.0])
            b_ub.append(-(M[i] + 1.0))
        
        # Bounds
        # S must be >= 1 (at least one hexagon)
        # t_x, t_y can be anything, but let's bound them to keep LP bounded.
        # The cluster is bounded, so t will be bounded by the optimal solution.
        # We can set wide bounds.
        bounds = [(-10, 10), (-10, 10), (1.0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                S_val = res.x[2]
                if S_val < best_S:
                    best_S = S_val
                    best_shape = shape
                    best_t = (res.x[0], res.x[1])
        except Exception:
            pass

    # If no shape was found (should not happen), fallback
    if best_shape is None:
        # Fallback to a simple line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the final inner hexagons
    # Apply the translation best_t to the best_shape
    inner = []
    for q, r in best_shape:
        x, y = axial_to_cartesian(q, r)
        x += best_t[0]
        y += best_t[1]
        inner.append([x, y, 0.0])
    
    # The outer hexagon is centered at (0,0) with side length best_S and angle 0.
    # We must ensure the inner hexagons are strictly inside or on the boundary.
    # The LP ensures dot(p, n) <= S*sqrt3/2 - 1.
    # This means the support of the inner hexagon is <= support of outer hexagon.
    # So they fit.
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
