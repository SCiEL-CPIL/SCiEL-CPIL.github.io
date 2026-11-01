"""Combined approach: Simulated Annealing with analytical container sizing and fast SAT."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Compute vertices of a regular hexagon."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def sat_disjoint(v1, v2):
        """
        Check if two convex polygons (hexagons) are disjoint using SAT.
        Returns True if disjoint, False if overlapping.
        """
        # SAT for two convex polygons
        # We need to check projection on normals of all edges of both polygons
        # v1, v2 are (6, 2) arrays
        
        # Pre-compute edge vectors and normals
        # For v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        
        # For v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        
        all_normals = np.vstack([normals1, normals2])
        
        # Normalize normals (SAT works with any direction, but normalization helps stability)
        norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_normals = all_normals / norms

        # Project vertices onto each normal
        # proj1: (12, 6), proj2: (12, 6)
        proj1 = v1.T @ all_normals # (2, 12) @ (12, 6) -> (2, 6) ? No.
        # v1 is (6, 2). all_normals is (12, 2).
        # We want dot product of each vertex with each normal.
        # (6, 2) @ (2, 12) -> (6, 12)
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # If there is any axis where the intervals [min, max] do not overlap, they are disjoint
        # Disjoint if max1 < min2 OR max2 < min1
        disjoint_mask = (max1 < min2) | (max2 < min1)
        
        return np.any(disjoint_mask)

    def compute_min_outer_side(inner, center, angle_deg):
        """
        Compute the minimum side length of a regular hexagon with given center and angle
        that contains all inner hexagons.
        """
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute vertices for all inner hexagons
        # This is the bottleneck, but we can optimize it if needed.
        # For now, just loop.
        all_verts = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts) # (6*N, 2)
        
        # Offset by center
        offsets = all_verts - np.array(center)
        
        # The outer hexagon has 6 edges. The distance from center to each edge is 'apothem' = side * sqrt(3)/2.
        # So side = 2 * apothem / sqrt(3).
        # The apothem must be at least the max projection of any inner vertex onto the normal of that edge.
        
        # Normals of the outer hexagon edges
        # The vertices are at angles angle + k*pi/3.
        # The edge normals are at angles angle + pi/6 + k*pi/3 (for flat-top) or similar.
        # Let's stick to the vertex definition.
        # If vertices are at theta, the edge between theta and theta+pi/3 has normal at theta + pi/6.
        
        for k in range(6):
            normal_angle = np.radians(angle_deg) + np.pi / 6 + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Project all inner vertices onto this normal
            projections = offsets @ normal
            max_proj = np.max(projections)
            
            if max_proj > max_extent:
                max_extent = max_proj
                
        # The required apothem is max_extent.
        # side = 2 * apothem / sqrt(3)
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """
        Objective function: returns the side length if valid, else a large penalty.
        params: [x1, y1, a1, ..., x12, y12, a12, cx, cy, ca]
        """
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+3]
        
        # 1. Check disjointness
        # Optimization: Only check pairs that are close?
        # For N=12, 66 pairs. SAT is fast enough if we don't do it every single time?
        # Let's try to optimize by pre-computing vertices only if needed, but SAT needs vertices.
        
        # We can speed up by checking center distance first.
        # Max distance for overlap is 2.0 (touching vertices of two side-1 hexagons).
        # If dist > 2.0, they are disjoint.
        
        centers = np.array([(inner[i][0], inner[i][1]) for i in range(N)])
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq <= 4.0: # 2.0^2
                    # Need precise check
                    v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                    if not sat_disjoint(v1, v2):
                        return 1e6
                # else: definitely disjoint, skip SAT
        
        # 2. Compute side
        side = compute_min_outer_side(inner, center, angle)
        
        return side

    def initial_guess():
        """
        Generate a good initial guess: Honeycomb packing.
        1 center, 6 around, 5 more in the second ring?
        Actually, 12 hexagons can be packed as a hexagon of side 2 (which has 1+6+12=19 spots) minus some?
        Or just a compact cluster.
        
        Let's use a simple grid-like honeycomb.
        Row 1: 2 hexagons
        Row 2: 3 hexagons
        Row 3: 3 hexagons
        Row 4: 2 hexagons
        Row 5: 2 hexagons?
        
        Better: 
        Center (0,0)
        Ring 1: 6 at distance 2, angles 0, 60, ...
        Ring 2: 5 at distance 4? No, that's too far.
        
        Let's try a "flower" shape:
        Center (0,0)
        6 neighbors at dist 2.
        Total 7.
        Need 5 more.
        Place them in the gaps of the outer ring?
        The gaps in ring 1 are at distance 2*sqrt(3) ~ 3.46, angles 30, 90, ...
        Place 5 of them there.
        """
        params = np.zeros(3*N + 4)
        positions = []
        
        # Center
        positions.append((0, 0, 0))
        
        # Ring 1 (6 hexagons)
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
            
        # Ring 2 (5 hexagons) - Place in the "valleys" of Ring 1
        # Distance to center for valley is 2 * sqrt(3)
        # Angle offset by 30 degrees
        for i in range(5):
            theta = np.pi/6 + i * np.pi / 3
            # Only 5, so we leave one spot empty to save space? Or just place 5.
            positions.append((2*np.sqrt(3)*np.cos(theta), 2*np.sqrt(3)*np.sin(theta), 0))
            
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
            
        # Container center and angle
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = 0.0
        
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    step_size = 0.5
    max_steps = 200000
    
    step_count = 0
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb a random parameter
        # We should probably perturb positions more often than angles or center
        idx = random.randint(0, 3*N + 3)
        
        # Don't perturb the side (it's calculated)
        # params[3*N+2] is unused/placeholder in our logic, but let's skip it anyway
        if idx == 3*N + 2:
             idx = 3*N + 3 # fallback to angle
            
        # Adaptive step size
        current_step = step_size * (0.5 + 0.5 * T)
        
        # For angles, use smaller steps?
        if idx % 3 == 2 or idx == 3*N + 3:
             current_step *= 0.1
            
        new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        # Acceptance criterion
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Extract best result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+3])
    side = compute_min_outer_side(inner, center, angle)
    
    # Final validation to ensure we don't return an invalid state
    # Check disjointness one last time
    valid = True
    centers = np.array([(inner[i][0], inner[i][1]) for i in range(N)])
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if dx*dx + dy*dy <= 4.0:
                v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not sat_disjoint(v1, v2):
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a very large container if something went wrong (should not happen)
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 100.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
