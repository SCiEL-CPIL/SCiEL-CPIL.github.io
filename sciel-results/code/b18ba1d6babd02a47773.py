# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    time_limit = 600.0
    n = 11

    # 1. Precompute geometry
    # Inner hexagons: Unit side, angle 0. Vertices at 0, 60, ...
    inner_angles_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles_rad])
    
    # Outer hexagon: Side S, angle 0. Vertices at 0, 60, ...
    # Normals to edges are at 30, 90, ...
    outer_normals_angles_rad = np.deg2rad([30, 90, 150, 210, 270, 330])
    outer_normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normals_angles_rad])
    
    # Minkowski sum of two unit hexagons (angle 0) is a regular hexagon with side 2.
    # Normals to the Minkowski sum edges are the same as the outer hexagon normals (30, 90, ...).
    # The apothem of the Minkowski sum (side 2) is sqrt(3).
    # Condition for non-overlap: For all pairs, the vector d = c_j - c_i must be outside the Minkowski sum.
    # This means: max_k dot(d, n_k) >= sqrt(3) OR min_k dot(d, n_k) <= -sqrt(3)?
    # Actually, the Minkowski sum is centered at 0. The condition for c2-c1 to be OUTSIDE is:
    # max_k dot(c2-c1, n_k) > apothem OR max_k dot(c1-c2, n_k) > apothem?
    # No, the Minkowski sum is symmetric. The condition is simply:
    # The vector d is outside the hexagon defined by |dot(d, n_k)| <= apothem for all k?
    # No, a regular hexagon defined by 6 half-planes is: dot(x, n_k) <= apothem for all k.
    # So x is INSIDE if dot(x, n_k) <= apothem for all k.
    # We want x to be OUTSIDE.
    # So we want: NOT (dot(x, n_k) <= apothem for all k).
    # Which is: EXISTS k such that dot(x, n_k) > apothem.
    # Wait, this is only true if the hexagon is the intersection of half-planes.
    # Yes, a convex polygon is the intersection of its half-planes.
    # So if d is inside, dot(d, n_k) <= apothem for ALL k.
    # If d is outside, dot(d, n_k) > apothem for SOME k.
    # So the condition for non-overlap is:
    # max_k dot(c_j - c_i, n_k) > sqrt(3) OR max_k dot(c_i - c_j, n_k) > sqrt(3)?
    # Since dot(c_i - c_j, n_k) = -dot(c_j - c_i, n_k), and the set of normals includes opposites (30 and 210),
    # max_k dot(d, n_k) is actually the support function h(d).
    # For a symmetric hexagon, h(d) = max_k |dot(d, n_k)|? No.
    # The normals are 30, 90, 150, 210, 270, 330.
    # n_0 = (cos30, sin30), n_3 = (cos210, sin210) = -n_0.
    # So dot(d, n_3) = -dot(d, n_0).
    # max(dot(d, n_0), dot(d, n_3)) = max(a, -a) = |a|.
    # So max_k dot(d, n_k) = max_k |dot(d, n_k)| for k in {0,1,2} (since 3,4,5 are opposites).
    # Let h(d) = max_{k in 0..5} dot(d, n_k).
    # The condition for d to be OUTSIDE the Minkowski sum (apothem A=sqrt(3)) is h(d) > A.
    # So for all pairs i, j: h(c_j - c_i) > sqrt(3).
    
    # Let's verify this.
    # If d = (0, 0), h(d) = 0. 0 < sqrt(3). Inside. Overlap. Correct.
    # If d = (2, 0).
    # dot((2,0), n_0) = 2 * cos(30) = sqrt(3).
    # dot((2,0), n_1) = 2 * cos(90) = 0.
    # h(d) = sqrt(3).
    # Condition h(d) > sqrt(3) is FALSE. So they touch?
    # The condition for non-overlap (strictly separated) is h(d) >= A.
    # The task says "touching or a smaller gap is an intersection".
    # So we need strict separation? "projected intervals are apart by more than 1e-6".
    # So we need h(d) > sqrt(3) + epsilon.
    
    # In the simulation, we will use a potential that penalizes h(d) < sqrt(3).
    
    def get_outer_side(centers, outer_center):
        # centers: (11, 2)
        # outer_center: (2,)
        # The outer hexagon must contain all inner vertices.
        # Inner vertices relative to outer center: v - outer_center.
        # Condition: For all vertices v, and all normals n_k:
        # dot(v - outer_center, n_k) <= apothem_outer
        # apothem_outer = S * sqrt(3) / 2.
        # So S >= max_{v, k} dot(v - outer_center, n_k) * 2 / sqrt(3).
        max_proj = 0.0
        for i in range(n):
            verts = inner_verts_rel + centers[i]
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
        return max_proj * 2 / np.sqrt(3)

    def check_validity(centers, outer_center, side):
        # Check non-overlap
        # h(c_j - c_i) > sqrt(3) + 1e-6
        for i in range(n):
            for j in range(i+1, n):
                d = centers[j] - centers[i]
                projs = d @ outer_normals.T
                if np.max(projs) < np.sqrt(3) + 1e-6:
                    return False
        # Check containment
        # All vertices inside outer hexagon
        # dot(v - outer_center, n_k) <= side * sqrt(3)/2 - 1e-6
        apothem = side * np.sqrt(3) / 2
        for i in range(n):
            verts = inner_verts_rel + centers[i]
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            if np.max(projs) > apothem + 1e-6:
                return False
        return True

    def _format_result(params):
        n = 11
        centers = params[:2*n].reshape(n, 2)
        outer_center = params[2*n:2*n+2]
        
        # Calculate side length
        max_proj = 0.0
        for i in range(n):
            verts = inner_verts_rel + centers[i]
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
            
        side = max_proj * 2 / np.sqrt(3)
        side += 1e-5 # Safety margin
        
        inner_hex_data = np.zeros((n, 3))
        inner_hex_data[:, 0] = centers[:, 0]
        inner_hex_data[:, 1] = centers[:, 1]
        inner_hex_data[:, 2] = 0.0
        
        outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
        
        return inner_hex_data, outer_hex_data, float(side)

    # Simulation setup
    # State: centers (11, 2), velocities (11, 2)
    # Forces:
    # 1. Repulsion between hexagons: If h(d) < sqrt(3), push apart.
    #    Gradient of h(d) w.r.t d is n_k where k is argmax.
    #    Force on i is -grad, on j is +grad.
    # 2. Containment: If vertex v is close to boundary, push in.
    #    This is effectively pushing the center towards the center of the outer hexagon.
    #    Or we can just minimize the side length by allowing the outer hexagon to shrink.
    #    Better: Fix the outer hexagon size S, and apply forces to keep hexagons inside.
    #    We want to minimize S. So we can start with a large S and gradually decrease it?
    #    Or use an adaptive S: S = max_proj * 2/sqrt(3) + margin.
    #    If we just simulate with fixed S, we find a packing for that S.
    #    Then we can binary search for the minimum S?
    #    Or, simulate with a "soft" boundary that pushes back, and the equilibrium S is the min S.
    #    Let's try: Start with S0. Simulate. If all inside, reduce S. If collision, increase S.
    #    This is slow.
    
    # Alternative: Objective function minimization via gradient descent on the physics state.
    # Let's stick to the mechanics simulation.
    # We will simulate for a fixed number of steps.
    # We will try multiple initial configurations and multiple S values?
    # No, let's simulate with a "spring" that tries to shrink the bounding box.
    # Force on center i: F_i = -k * (v_i - outer_center) ? No, that's a central force.
    # We want to minimize the max projection.
    # This is equivalent to minimizing the "diameter" in the hexagonal metric.
    
    # Let's use a penalty method within the simulation.
    # Potential U = U_collision + U_containment.
    # U_collision = sum_{i<j} max(0, sqrt(3) - h(d_ij))^2.
    # U_containment = sum_{i, v} max(0, dot(v - c_out, n_k) - apothem)^2.
    # But we want to MINIMIZE apothem (or S).
    # So we can't just minimize U. We want to find the smallest S such that U_collision = 0 and U_containment = 0.
    
    # Strategy:
    # 1. Generate initial cluster.
    # 2. Set S to a value that is definitely large enough (e.g., 10).
    # 3. Simulate with fixed S, allowing hexagons to move freely but repelling each other.
    #    This will relax them into a valid packing for that S.
    # 4. Once relaxed (forces ~ 0), calculate the actual min S required.
    # 5. If min_S < S, reduce S and repeat?
    #    Or, better: Use the simulation to find a valid packing, then refine with local search.
    
    # Let's implement a robust simulation.
    
    def simulate(centers_init, S_target, steps=5000, dt=0.01, damping=0.99):
        centers = centers_init.copy()
        velocities = np.zeros_like(centers)
        
        apothem = S_target * np.sqrt(3) / 2
        
        for step in range(steps):
            forces = np.zeros_like(centers)
            
            # Collision forces
            for i in range(n):
                for j in range(i+1, n):
                    d = centers[j] - centers[i]
                    projs = d @ outer_normals.T
                    max_proj = np.max(projs)
                    idx = np.argmax(projs)
                    n_k = outer_normals[idx]
                    
                    # Check if inside Minkowski sum
                    if max_proj < np.sqrt(3):
                        # Penalty
                        # Gradient of (sqrt(3) - max_proj)^2 w.r.t d
                        # d/dd (sqrt(3) - max(d, n_k))^2 = -2 * (sqrt(3) - max_proj) * n_k
                        # Force on j is +grad, on i is -grad.
                        # We want to push them apart, so force on j is in direction n_k, on i is -n_k.
                        penalty = np.sqrt(3) - max_proj
                        f = 10.0 * penalty * n_k
                        forces[j] += f
                        forces[i] -= f
            
            # Containment forces
            # For each hexagon, check vertices.
            for i in range(n):
                verts = inner_verts_rel + centers[i]
                verts_shifted = verts - np.zeros(2) # Outer center is 0 for now? 
                # We should allow outer center to move? 
                # The problem allows outer center to be anywhere.
                # But we can fix it to (0,0) by symmetry? 
                # No, the cluster might be off-center.
                # However, we can always translate the whole system so the center of mass is 0.
                # And the outer hexagon can be centered at the center of mass.
                # So let's fix outer_center = (0,0) and just ensure the packing fits in a hexagon centered at (0,0).
                # This simplifies things. We don't need to optimize outer_center.
                
                # So verts_shifted = verts.
                projs = verts @ outer_normals.T
                for k in range(6):
                    if projs[k] > apothem:
                        # Push center i in direction -n_k
                        penalty = projs[k] - apothem
                        f = 10.0 * penalty * outer_normals[k]
                        forces[i] -= f
            
            # Update velocities and positions
            velocities = velocities * damping + forces * dt
            centers = centers + velocities * dt
            
            # Optional: Keep center of mass at 0
            com = np.mean(centers, axis=0)
            centers -= com
            velocities -= com
            
        return centers

    # Initial configurations
    configs = []
    
    # 1. Hexagonal grid cluster
    # 1 center, 6 neighbors, 4 more?
    # Hex grid coordinates (axial):
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) -> 6 neighbors
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) -> 5 more. Total 12. Remove one.
    # Let's just place them randomly in a circle.
    
    np.random.seed(42)
    
    # Config 1: Random
    c1 = np.random.rand(n, 2) * 4 - 2
    
    # Config 2: Ring
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n) * 2.0
    c2 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    c2[0] = [0,0]
    
    # Config 3: Grid
    # Approximate hex grid
    c3 = np.array([
        [0, 0],
        [1.732, 0], [0.866, 1.5], [-0.866, 1.5], [-1.732, 0], [-0.866, -1.5], [0.866, -1.5],
        [3.464, 0], [2.598, 1.5], [0, 3.0], [-2.598, 1.5], [-3.464, 0]
    ])
    
    configs = [c1, c2, c3]
    
    best_side = float('inf')
    best_params = None
    
    for config in configs:
        # Start with large S
        S = 10.0
        # Simulate
        centers = simulate(config, S, steps=2000)
        
        # Calculate actual min S
        actual_S = get_outer_side(centers, np.zeros(2))
        
        # Refine: Simulate with S = actual_S * 1.01
        S = actual_S * 1.01
        centers = simulate(centers, S, steps=2000)
        actual_S = get_outer_side(centers, np.zeros(2))
        
        if actual_S < best_side:
            best_side = actual_S
            best_params = np.concatenate([centers.flatten(), [0,0]])
            
        if time.time() - start_time > time_limit - 60:
            break
            
    # Local refinement using Nelder-Mead on the final positions
    # This helps to squeeze out the last bit of space.
    
    def objective(params):
        centers = params[:2*n].reshape(n, 2)
        side = get_outer_side(centers, np.zeros(2))
        
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = centers[j] - centers[i]
                projs = d @ outer_normals.T
                max_proj = np.max(projs)
                if max_proj < np.sqrt(3) + 1e-6:
                    penalty += (np.sqrt(3) + 1e-6 - max_proj)**2
                    
        return side + 100 * penalty
    
    if best_params is not None:
        try:
            from scipy.optimize import minimize
            res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < objective(best_params):
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        best_params = np.concatenate([configs[0].flatten(), [0,0]])
        
    return _format_result(best_params)

# EVOLVE-BLOCK-END
