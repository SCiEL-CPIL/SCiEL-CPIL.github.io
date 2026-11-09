"""
Strategy: Constraint-Aware Local Refinement with Bisection on L.

The current approach uses Differential Evolution (DE) on 36 variables (12 centers, 12 angles).
This is inefficient because:
1. DE explores a vast space where most configurations are infeasible (overlapping).
2. The objective function (required L) is piecewise linear but non-smooth, making gradient-based or
   directed local search difficult without careful handling.
3. DE is slow to converge for tight packing problems.

New Strategy:
1. Fix the outer hexagon center at (0,0) and rotation at 0 (symmetry argument: optimal packing
   of symmetric objects in a symmetric container often aligns symmetries, or we can fix one
   to reduce degrees of freedom; actually, we can allow outer rotation but fix center to 0,0
   due to translation invariance. Let's keep outer rotation fixed at 0 for simplicity in the
   first pass, as the inner hexagons can rotate freely. If needed, we could add outer rotation
   as a variable, but 12*3 + 1 = 37 vars is similar. Let's stick to 36 vars: 12*(x,y,angle).)
   
   Wait, the interface allows outer rotation. However, for a regular hexagon container,
   rotating the container by 30 degrees maps it to itself (if we consider the set of constraints).
   Actually, a regular hexagon has 6-fold rotational symmetry. Rotating the container by 60 degrees
   is equivalent to rotating the inner objects by -60 degrees relative to the container.
   Since inner objects can rotate freely, the outer rotation angle is redundant if we allow
   inner rotations. We can fix outer_angle = 0.0 without loss of generality for the optimal L,
   provided the inner angles are free.
   
   So, we optimize 36 variables: x_i, y_i, theta_i for i=1..12.

2. Objective: Minimize L such that all inner hexagons are inside the outer hexagon of side L
   centered at origin, and no two inner hexagons overlap.

3. Feasibility Check for a given L:
   - For each inner hexagon i, check if it fits in the outer hexagon of side L.
     This is equivalent to: max_{v in verts(i)} support(v, outer_normals) <= L * sqrt(3)/2.
     Or more simply, compute the required L_i for each inner hexagon i to fit alone, and
     L = max(L_i).
   - Check for overlaps between all pairs (i, j).
     Two convex polygons overlap if and only if there exists a separating axis.
     For regular hexagons, the potential separating axes are the normals to the edges of
     either polygon. There are 6 normals for hex i and 6 for hex j. Total 12 axes.
     We can check separation on these axes. If they are separated on any axis, they don't overlap.
     If they are not separated on any axis, they overlap.
     Actually, for convex polygons, if the projections on all edge normals of both polygons
     overlap, then the polygons intersect.
     
4. Optimization Loop:
   - Start with a valid, loose configuration (e.g., hexagonal grid or circle).
   - Use a local search method (e.g., Nelder-Mead or a custom gradient-free local search)
     to minimize a penalty function:
       Penalty = max(0, L_current - L_target) + sum(overlap_penalties)
     But this is still black-box.
     
   Better: Use a "Pushing" algorithm or "Jiggling" algorithm.
   - Compute the "required L" for the current configuration.
   - If L is too large, try to move vertices/centers to reduce the max support.
   - If overlaps exist, try to separate them.
   
   Alternative: Use `scipy.optimize.minimize` with a composite objective that strongly penalizes
   overlaps and infeasibility, starting from multiple good initial guesses.
   
   Let's try a specific structural change: **Simulated Annealing** or **Basin Hopping** 
   with a much faster feasibility check and objective evaluation.
   
   Key Insight for Speed:
   - The overlap check can be vectorized or optimized.
   - The required L calculation is fast.
   
   Let's implement a **Basin Hopping** approach with a high-quality local minimizer.
   The local minimizer will minimize the "violation" function:
   V(x) = max(0, L_config - L_min) + lambda * sum(overlap_depths)
   But we want to minimize L directly.
   
   Let's define the objective as:
   f(x) = L_config(x) + C * max(0, overlap_violation(x))
   where overlap_violation is the minimum separation needed to resolve overlaps.
   Computing minimum separation is hard.
   
   Simpler:
   f(x) = L_config(x) + C * num_overlaps(x)
   This is non-smooth but works with Nelder-Mead.
   
   To improve on the current DE approach:
   1. Use a better initial guess (e.g., optimal packing for smaller n, or random + local polish).
   2. Use `basinhopping` from scipy, which is designed for global optimization of local minima.
   3. The local step uses Nelder-Mead on a smoothed objective.
   
   Let's refine the overlap check to be as fast as possible.
   Pre-compute normals.
   
   Also, we can exploit symmetry. The problem is symmetric under 60-degree rotation.
   We can fix one hexagon's angle to 0 (or constrain the center of mass to 0,0).
   Actually, fixing the center of mass to (0,0) is a good constraint to reduce drift.
   
   Let's try this structure:
   1. Generate diverse initial configurations.
   2. For each, run a local minimization (Nelder-Mead) to minimize:
      f(x) = L_required(x) + 1000 * (1 if any overlap else 0)
      This encourages finding a valid configuration with small L.
   3. Among valid results, keep the one with smallest L.
   4. Use Basin Hopping to jump between basins.

   Wait, the current program already does DE + NM. The score is 6.6.
   The theoretical minimum for 12 unit hexagons in a hexagon?
   For n=1, L=1.
   For n=7 (1 center, 6 around), L=3.
   For n=12, it's denser.
   
   Let's look at the "Inspirations". They didn't provide code, just notes.
   
   What if we use a **Grid Search + Local Refinement** hybrid?
   No, 36D grid is impossible.
   
   Let's try **Differential Evolution with Constraints** handled via penalty, but with a 
   much tighter bound and faster convergence, or **CMA-ES** (if available in scipy? No, 
   `scipy.optimize` has `differential_evolution`, `minimize`, `basinhopping`).
   
   Let's try **Basin Hopping** with a specific local minimizer that handles the non-smoothness
   better.
   
   Actually, a different family: **Geometric Construction + Fine Tuning**.
   We can start with a known good packing pattern (e.g., 3 rows of 4, or a hexagonal cluster)
   and then use local search to tighten it.
   
   Let's implement a **Multi-Start Local Search** with **Simulated Annealing** style acceptance
   for the local steps to escape local minima, but constrained to stay feasible or near-feasible.
   
   However, the most robust "structural" change from DE is to use **Basin Hopping** which
   explicitly searches for global minima by hopping between basins, using a local minimizer
   (Nelder-Mead) for the "basin" part. DE is also a global search, but Basin Hopping is
   often more effective for these types of "energy landscape" problems if the local minimizer
   is good.
   
   Let's optimize the `required_L` and `check_overlap` functions for speed using NumPy.
   
   Vectorized Overlap Check:
   For a pair (i, j), we need to check if their projections on 12 axes overlap.
   Axes are normals of i and j.
   Normal of hex k at angle theta_k: angles are theta_k + 30 + i*60.
   
   Let's write the code.
"""

import time
import math
import numpy as np
from scipy.optimize import basinhopping, minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon as a numpy array."""
        angle_rad = math.radians(angle_deg)
        # Vertices at angle_rad + i*pi/3
        thetas = angle_rad + np.arange(6) * (math.pi / 3.0)
        xs = cx + side * np.cos(thetas)
        ys = cy + side * np.sin(thetas)
        return np.column_stack((xs, ys))

    def get_hex_normals(angle_deg):
        """Get the 6 outward normals of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        # Normals at angle_rad + 30 + i*60
        thetas = angle_rad + math.radians(30.0) + np.arange(6) * (math.pi / 3.0)
        ns = np.column_stack((np.cos(thetas), np.sin(thetas)))
        return ns

    def compute_required_L(centers, angles, outer_angle=0.0):
        """
        Compute the minimum side length L of the outer hexagon (centered at 0,0, rotated by outer_angle)
        to enclose all inner hexagons.
        """
        outer_angle_rad = math.radians(outer_angle)
        # Outer hexagon normals (fixed at 0 rotation for simplicity in this function, 
        # but we can pass outer_angle if needed. Here we assume outer is aligned with axes 
        # or we just compute the max projection on the 6 fixed axes of the outer hexagon)
        
        # For a regular hexagon centered at origin, the condition for a point p to be inside
        # with side L is: |dot(p, n_k)| <= L * sqrt(3) / 2 for all 6 normals n_k?
        # No, the hexagon is the intersection of half-planes dot(p, n_k) <= apothem.
        # The normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        outer_normals = []
        for i in range(6):
            theta = math.radians(30.0 + i * 60.0)
            outer_normals.append([math.cos(theta), math.sin(theta)])
        outer_normals = np.array(outer_normals) # shape (6, 2)
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            # Rotate vertices by -outer_angle if outer is rotated
            if outer_angle != 0.0:
                c = math.cos(-outer_angle_rad)
                s = math.sin(-outer_angle_rad)
                rot_mat = np.array([[c, -s], [s, c]])
                verts = verts @ rot_mat.T
            
            # Project all vertices onto outer normals
            # proj shape: (6, 6) -> 6 vertices, 6 normals
            projs = verts @ outer_normals.T
            # We need the max positive projection
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
        
        # L * sqrt(3) / 2 >= max_proj  => L >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / SQRT3

    def check_overlap_fast(centers, angles):
        """
        Check if any two hexagons overlap.
        Returns (is_overlap, min_separation) where min_separation is the negative of the 
        maximum penetration depth (if overlapping) or the minimum distance (if not).
        For optimization, we just need to know if they overlap and by how much to penalize.
        
        Two convex polygons A and B are disjoint if there exists an axis u such that
        proj_A(u) and proj_B(u) are disjoint.
        The candidate axes are the normals of A and B.
        """
        n = len(centers)
        total_violation = 0.0
        is_overlap = False
        
        for i in range(n):
            verts_i = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            normals_i = get_hex_normals(angles[i])
            
            for j in range(i + 1, n):
                verts_j = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                normals_j = get_hex_normals(angles[j])
                
                # Combine normals: 12 axes
                all_normals = np.vstack((normals_i, normals_j))
                
                # Project vertices of i and j onto all normals
                # proj_i: (12, 6) -> 12 axes, 6 vertices
                proj_i = verts_i @ all_normals.T
                proj_j = verts_j @ all_normals.T
                
                # Min and max projections for each axis
                min_i = np.min(proj_i, axis=1)
                max_i = np.max(proj_i, axis=1)
                min_j = np.min(proj_j, axis=1)
                max_j = np.max(proj_j, axis=1)
                
                # Check separation on each axis
                # If max_i < min_j or max_j < min_i, they are separated on this axis.
                # If they are separated on ANY axis, they are disjoint.
                separated = np.any((max_i < min_j) | (max_j < min_i))
                
                if not separated:
                    is_overlap = True
                    # Calculate penetration depth for penalty
                    # The penetration depth is the minimum amount needed to separate them.
                    # It is the minimum over all axes of the overlap length.
                    overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                    # overlaps will be positive where they overlap.
                    # We want the minimum positive overlap? No, the separation distance is 
                    # the minimum over axes of the "gap" (which is negative if overlapping).
                    # Actually, for penalty, we can sum the overlaps or take the max.
                    # Let's use the minimum separation distance (negative of max penetration).
                    # sep = min_j (min_j - max_i) for each axis?
                    # If they overlap on an axis, the "distance" is negative.
                    # dist_axis = min_j - max_i (if i is to the left of j)
                    # We need the direction.
                    # Simpler penalty: sum of positive overlaps on all axes.
                    total_violation += np.sum(np.maximum(0, overlaps))
                    
        return is_overlap, total_violation

    def objective(params):
        """
        Objective function for optimization.
        params: 36 floats (x1, y1, a1, ..., x12, y12, a12)
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[2*i], params[2*i+1]))
            angles.append(params[2*N + i])
            
        L = compute_required_L(centers, angles, 0.0)
        is_overlap, violation = check_overlap_fast(centers, angles)
        
        # Penalty for overlap
        # We want to minimize L, but if there is overlap, it's invalid.
        # Use a large penalty proportional to violation.
        penalty = 100.0 * violation if is_overlap else 0.0
        
        return L + penalty

    def generate_initial_guesses(num_guesses=20):
        """Generate diverse initial configurations."""
        guesses = []
        
        # Guess 1: Hexagonal grid
        # 12 hexagons.
        # Rows of 3, 4, 3, 2?
        # Let's try a simple circular arrangement
        for k in range(num_guesses):
            centers = []
            angles = []
            # Random radius and angle
            r = 1.5 + 0.5 * k
            for i in range(N):
                theta = 2 * math.pi * i / N + k * 0.1
                cx = r * math.cos(theta)
                cy = r * math.sin(theta)
                centers.append((cx, cy))
                angles.append(0.0)
            
            params = []
            for i in range(N):
                params.append(centers[i][0])
                params.append(centers[i][1])
            for i in range(N):
                params.append(angles[i])
            guesses.append(params)
            
        # Guess 2: Tighter grid
        # 2 rows of 6?
        # Hex packing:
        # Row 0: y=0, x = 0, 2, 4, 6, 8, 10
        # Row 1: y=sqrt(3), x = 1, 3, 5, 7, 9
        # Total 11. Need 12.
        # Let's just use random + local polish as the primary driver.
        
        return guesses

    best_L = float('inf')
    best_params = None

    initial_guesses = generate_initial_guesses(10)
    
    # Run Basin Hopping for each initial guess
    # Basin Hopping uses a local minimizer (Nelder-Mead) and then hops.
    
    bounds = [(-5.0, 5.0)] * (2*N) + [(0.0, 180.0)] * N
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Define the local minimizer step
            def local_min(x):
                # Nelder-Mead
                res = minimize(objective, x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                return res.x, res.fun, 0
                
            # Run basinhopping
            # niter=10 might be enough if local is good
            # stepsize=0.5
            bh_res = basinhopping(
                objective, 
                x0, 
                minimizer_kwargs={'options': {'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}},
                niter=10, 
                stepsize=0.5,
                seed=42
            )
            
            if bh_res.fun < best_L:
                # Check validity
                centers = []
                angles = []
                for j in range(N):
                    centers.append((bh_res.x[2*j], bh_res.x[2*j+1]))
                    angles.append(bh_res.x[2*N + j])
                
                is_overlap, _ = check_overlap_fast(centers, angles)
                if not is_overlap:
                    L_val = compute_required_L(centers, angles, 0.0)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = bh_res.x
                        
        except Exception as e:
            pass

    # If Basin Hopping didn't find a good solution, try a few random starts with just NM
    if best_params is None or best_L > 10.0:
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            x0 = np.random.uniform(-3, 3, 2*N) + np.zeros(N)
            x0 = np.concatenate((x0, np.random.uniform(0, 180, N)))
            
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_L:
                centers = []
                angles = []
                for j in range(N):
                    centers.append((res.x[2*j], res.x[2*j+1]))
                    angles.append(res.x[2*N + j])
                
                is_overlap, _ = check_overlap_fast(centers, angles)
                if not is_overlap:
                    L_val = compute_required_L(centers, angles, 0.0)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x

    # Final fallback if no valid solution found
    if best_params is None:
        # Return a valid, loose configuration
        # 3x4 grid
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0
                y = r * SQRT3
                inner.append([x, y, 0.0])
        # Center it
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in inner:
            p[0] -= cx
            p[1] -= cy
        
        L_val = compute_required_L([(p[0], p[1]) for p in inner], [p[2] for p in inner], 0.0)
        best_L = L_val
        best_params = [p[0] for p in inner] + [p[1] for p in inner] + [p[2] for p in inner]

    # Construct return value
    centers = []
    angles = []
    for i in range(N):
        centers.append([best_params[2*i], best_params[2*i+1]])
        angles.append(best_params[2*N + i])
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    final_L = compute_required_L([(c[0], c[1]) for c in centers], angles, 0.0)
    
    # Verify validity one last time
    is_overlap, _ = check_overlap_fast([(c[0], c[1]) for c in centers], angles)
    if is_overlap:
        # If somehow invalid, return a known valid loose config
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        final_L = 40.0

    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
