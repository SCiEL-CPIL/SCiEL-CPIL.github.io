"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# Precompute hexagon vertices for a unit regular hexagon (side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
])

# For SAT with two convex polygons, we need the normal vectors of each edge
# For a regular hexagon, the edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
_HEX_NORMALS = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(30, 360, 60)
])


def _get_transformed_hex(cx, cy, angle_deg, scale=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation and scale."""
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    verts = scale * (_HEX_VERTS @ R.T) + np.array([cx, cy])
    return verts


def _sat_min_separation(verts_a, verts_b, normals_a, normals_b):
    """
    Compute the minimum translation distance to separate two convex polygons using SAT.
    Returns the minimum distance along any separating axis (0 if overlapping or touching).
    Actually, we want the minimum distance to move one polygon so they don't overlap.
    
    For two convex polygons, the minimum separation distance is the maximum over all 
    candidate axes of the required gap along that axis.
    
    Let me think about this differently. For SAT:
    - Project both polygons onto each candidate axis (normals of both polygons)
    - If there's a gap on any axis, the polygons are separated
    - The minimum distance to separate them is the maximum over all axes of the 
      "overlap amount" (negative gap) on that axis
    
    Actually, the correct approach:
    For each axis, compute the projection intervals of both polygons.
    If they overlap on all axes, the polygons overlap.
    The minimum translation to separate is the maximum over all axes of the 
    overlap depth on that axis.
    """
    min_dist = 0.0
    for axis in np.vstack([normals_a, normals_b]):
        # Project both polygons onto this axis
        proj_a = np.dot(verts_a, axis)
        proj_b = np.dot(verts_b, axis)
        
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()
        
        # Check for separation on this axis
        # Gap = min(max_a, max_b) - max(min_a, min_b)
        # If gap > 0, separated on this axis
        # If gap <= 0, overlap by -gap on this axis
        gap = min(max_a, max_b) - max(min_a, min_b)
        
        if gap < 0:
            # Overlap on this axis by -gap
            min_dist = max(min_dist, -gap)
    
    return min_dist


def _outer_hex_contains_inner(outer_center, outer_side, outer_angle, 
                              inner_center, inner_angle):
    """
    Check if a unit hexagon at inner_center with inner_angle is fully contained 
    in the outer hexagon.
    
    The outer hexagon has side length `outer_side` and is rotated by `outer_angle`.
    
    For a regular hexagon with side length s, the inradius is s * sqrt(3)/2.
    The distance from center to each edge is the inradius.
    
    A point is inside the outer hexagon if it is on the correct side of each edge.
    For the inner hexagon to be fully inside, all its vertices must be inside.
    
    But we need to ensure the entire inner hexagon is inside, not just vertices.
    Since both are convex, if all vertices of the inner are inside the outer, 
    the inner is inside the outer.
    
    Actually, that's not quite right. We need to check that the inner hexagon 
    doesn't cross any edge of the outer hexagon.
    
    Better approach: For each edge of the outer hexagon, check that all vertices 
    of the inner hexagon are on the interior side.
    """
    outer_verts = _get_transformed_hex(outer_center[0], outer_center[1], 
                                       outer_angle, scale=outer_side)
    inner_verts = _get_transformed_hex(inner_center[0], inner_center[1], 
                                       inner_angle, scale=1.0)
    
    # Get outer hexagon edge normals (pointing inward)
    # For a regular hexagon, the inward normals are at angles 30+180, 90+180, etc.
    # Actually, let's compute them from the vertices
    n_outer = len(outer_verts)
    for i in range(n_outer):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % n_outer]
        edge = v2 - v1
        # Inward normal: rotate edge by -90 degrees and normalize
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Check that all inner vertices are on the interior side
        # The interior is the side where the center is
        center_to_v1 = outer_center - v1
        if np.dot(center_to_v1, normal) < 0:
            normal = -normal
        
        for iv in inner_verts:
            dist = np.dot(iv - v1, normal)
            if dist < -1e-9:
                return False
    
    return True


def _max_required_side(inner_configs, outer_center, outer_angle):
    """
    Given a set of inner hexagon configurations and outer hexagon center/angle,
    find the minimum outer side length such that all inner hexagons fit inside.
    
    We binary search on the outer side length, or we can compute it directly.
    
    Actually, for a fixed outer center and angle, the minimum outer side length 
    is determined by the farthest inner hexagon vertex from the center in terms 
    of the hexagonal distance metric.
    
    Let me think differently. For a regular hexagon centered at C with angle A,
    a point P is inside if for each of the 6 edges, P is on the interior side.
    The minimum side length s such that all inner vertices are inside can be 
    computed by checking each inner vertex against each outer edge direction.
    
    For a regular hexagon with center C, angle A, and side s, the half-width 
    in each of the 6 normal directions is s * sqrt(3)/2.
    
    So for each inner vertex P, we need:
    For each of the 6 normal directions n_i (at angles A+30, A+90, A+150, A+210, A+270, A+330):
    dot(P - C, n_i) <= s * sqrt(3)/2
    
    So s >= max over all inner vertices P and all 6 normals n_i of:
    2 * dot(P - C, n_i) / sqrt(3)
    
    And s must be >= 0.
    """
    # Outer hexagon normal directions
    base_angles = np.radians([30, 90, 150, 210, 270, 330])
    outer_normals = np.array([
        [np.cos(base_angles[i] + np.radians(outer_angle)), 
         np.sin(base_angles[i] + np.radians(outer_angle))]
        for i in range(6)
    ])
    
    max_val = 0.0
    for ix, iy, iangle in inner_configs:
        inner_verts = _get_transformed_hex(ix, iy, iangle, scale=1.0)
        for iv in inner_verts:
            diff = iv - np.array(outer_center)
            projections = np.dot(outer_normals, diff)
            for proj in projections:
                if proj > max_val:
                    max_val = proj
    
    # s * sqrt(3)/2 >= max_val => s >= 2 * max_val / sqrt(3)
    return 2.0 * max_val / np.sqrt(3.0)


def _check_validity(inner_configs, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer."""
    # Check overlaps between inner hexagons
    n = len(inner_configs)
    for i in range(n):
        ix, iy, iangle = inner_configs[i]
        verts_i = _get_transformed_hex(ix, iy, iangle, scale=1.0)
        normals_i = _HEX_NORMALS @ np.array([[np.cos(np.radians(iangle)), -np.sin(np.radians(iangle))],
                                              [np.sin(np.radians(iangle)), np.cos(np.radians(iangle))]]).T
        for j in range(i + 1, n):
            jx, jy, jangle = inner_configs[j]
            verts_j = _get_transformed_hex(jx, jy, jangle, scale=1.0)
            normals_j = _HEX_NORMALS @ np.array([[np.cos(np.radians(jangle)), -np.sin(np.radians(jangle))],
                                                  [np.sin(np.radians(jangle)), np.cos(np.radians(jangle))]]).T
            sep = _sat_min_separation(verts_i, verts_j, normals_i, normals_j)
            if sep < -1e-9:
                return False, sep
    
    # Check all inner hexagons are inside outer
    for ix, iy, iangle in inner_configs:
        if not _outer_hex_contains_inner(outer_center, outer_side, outer_angle, 
                                          (ix, iy), iangle):
            return False, 0.0
    
    return True, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_side = 40.0
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Simulated Annealing
    # State: positions (x,y) and angles for 12 hexagons, plus outer center and angle
    # That's 12*3 + 2 + 1 = 39 parameters
    
    def _energy(state):
        """
        Compute the energy: maximum of (required outer side length, penalty for overlaps)
        """
        inner_configs = [[state[3*i], state[3*i+1], state[3*i+2]] for i in range(N)]
        outer_center = (state[36], state[37])
        outer_angle = state[38]
        
        # Compute required outer side
        req_side = _max_required_side(inner_configs, outer_center, outer_angle)
        
        # Check overlaps - add penalty
        penalty = 0.0
        for i in range(N):
            ix, iy, iangle = inner_configs[i]
            verts_i = _get_transformed_hex(ix, iy, iangle, scale=1.0)
            normals_i = _HEX_NORMALS @ np.array([[np.cos(np.radians(iangle)), -np.sin(np.radians(iangle))],
                                                  [np.sin(np.radians(iangle)), np.cos(np.radians(iangle))]]).T
            for j in range(i + 1, N):
                jx, jy, jangle = inner_configs[j]
                verts_j = _get_transformed_hex(jx, jy, jangle, scale=1.0)
                normals_j = _HEX_NORMALS @ np.array([[np.cos(np.radians(jangle)), -np.sin(np.radians(jangle))],
                                                      [np.sin(np.radians(jangle)), np.cos(np.radians(jangle))]]).T
                sep = _sat_min_separation(verts_i, verts_j, normals_i, normals_j)
                if sep < 0:
                    penalty += (-sep) ** 2
        
        return req_side + 10.0 * penalty
    
    # Initial state: spread out hexagons
    def _random_state():
        state = np.zeros(39)
        for i in range(N):
            state[3*i] = np.random.uniform(-5, 5)
            state[3*i+1] = np.random.uniform(-5, 5)
            state[3*i+2] = np.random.uniform(0, 60)
        state[36] = np.random.uniform(-2, 2)
        state[37] = np.random.uniform(-2, 2)
        state[38] = np.random.uniform(0, 60)
        return state
    
    # Try a few random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        state = _random_state()
        current_energy = _energy(state)
        
        # SA parameters
        T = 2.0
        T_min = 0.001
        cooling = 0.995
        n_iterations = 5000
        
        for it in range(n_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb a random subset of parameters
            n_perturb = np.random.randint(1, 5)
            indices = np.random.choice(39, n_perturb, replace=False)
            new_state = state.copy()
            step_size = 0.5 * (T / T_min) ** 0.1
            for idx in indices:
                new_state[idx] += np.random.normal(0, step_size)
            
            new_energy = _energy(new_state)
            
            # Accept or reject
            if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / T):
                state = new_state
                current_energy = new_energy
                
                # Check if this is a valid solution
                inner_configs = [[state[3*i], state[3*i+1], state[3*i+2]] for i in range(N)]
                outer_center = (state[36], state[37])
                outer_angle = state[38]
                req_side = _max_required_side(inner_configs, outer_center, outer_angle)
                
                # Quick overlap check
                valid = True
                for i in range(N):
                    ix, iy, iangle = inner_configs[i]
                    verts_i = _get_transformed_hex(ix, iy, iangle, scale=1.0)
                    normals_i = _HEX_NORMALS @ np.array([[np.cos(np.radians(iangle)), -np.sin(np.radians(iangle))],
                                                          [np.sin(np.radians(iangle)), np.cos(np.radians(iangle))]]).T
                    for j in range(i + 1, N):
                        jx, jy, jangle = inner_configs[j]
                        verts_j = _get_transformed_hex(jx, jy, jangle, scale=1.0)
                        normals_j = _HEX_NORMALS @ np.array([[np.cos(np.radians(jangle)), -np.sin(np.radians(jangle))],
                                                              [np.sin(np.radians(jangle)), np.cos(np.radians(jangle))]]).T
                        sep = _sat_min_separation(verts_i, verts_j, normals_i, normals_j)
                        if sep < -1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and req_side < best_side:
                    best_side = req_side
                    best_inner = inner_configs
                    best_outer_center = outer_center
                    best_outer_angle = outer_angle
            
            T = max(T * cooling, T_min)
    
    # Return best found
    if best_inner is None:
        # Fallback to original
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_outer_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
