import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute vertices for unit hexagons at various angles
    # For each angle, store the 6 local vertices
    angles_list = [0, 30, 60, 90, 120, 150]
    local_verts = {}
    for ang in angles_list:
        rad = np.radians(ang)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        base = np.array([
            [1.0, 0.0],
            [0.5, SQRT3/2.0],
            [-0.5, SQRT3/2.0],
            [-1.0, 0.0],
            [-0.5, -SQRT3/2.0],
            [0.5, -SQRT3/2.0]
        ])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local_verts[ang] = base @ rot.T

    def get_vertices(cx, cy, ang_idx):
        """Get vertices of a unit hexagon at (cx, cy) with angle index ang_idx."""
        ang = angles_list[ang_idx]
        return local_verts[ang] + np.array([cx, cy])

    def sat_disjoint(v1, v2, eps=1e-7):
        """Check if two convex polygons are disjoint using SAT."""
        polys = [v1, v2]
        for p in polys:
            n = len(p)
            for i in range(n):
                j = (i + 1) % n
                edge = p[j] - p[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
                    return True
        return False

    def is_valid(positions, angle_indices):
        """Check if all hexagons are pairwise disjoint."""
        verts_list = []
        for i in range(N):
            verts_list.append(get_vertices(positions[i], angle_indices[i]))
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    def outer_side_length(positions, angle_indices):
        """Calculate the minimum outer hexagon side length (centered at origin, angle 0)."""
        # We can also optimize the center and outer angle, but for now
        # let's fix outer to be centered at the centroid and angle 0,
        # and see if we can improve by allowing center/angle optimization later.
        # Actually, the score is just the side length. The evaluator likely
        # checks if all inner hexagons fit inside the outer one.
        # The task says "outer hexagon center (x, y), side length and rotation".
        # We should return the optimal outer hexagon parameters as well.
        
        # For a given set of inner hexagons, the minimal enclosing regular hexagon
        # is a non-trivial optimization problem.
        # However, a good approximation is to center it at the centroid of the
        # "bounding box" or centroid of all vertices, and try a few angles.
        
        # Let's collect all vertices
        all_verts = []
        for i in range(N):
            all_verts.append(get_vertices(positions[i], angle_indices[i]))
        all_verts = np.vstack(all_verts)
        
        # Try to find the best outer hexagon
        # We can parameterize the outer hexagon by center (cx, cy) and angle theta.
        # Side length S must be such that all vertices are inside.
        # For a regular hexagon of side S centered at (cx, cy) with angle theta,
        # the condition for a point (x, y) to be inside is:
        # The 6 support functions must be satisfied.
        
        # Let's use a simple grid search for the outer hexagon parameters
        # or a local optimization.
        
        # To keep it fast, we'll try a few candidate outer hexagons:
        # 1. Centered at (0,0), angle 0
        # 2. Centered at (0,0), angle 30
        # 3. Centered at centroid of vertices, angle 0
        # 4. Centered at centroid of vertices, angle 30
        
        centroid = np.mean(all_verts, axis=0)
        
        best_S = float('inf')
        best_cx = 0.0
        best_cy = 0.0
        best_angle = 0.0
        
        # Candidate centers
        centers = [(0.0, 0.0), (centroid[0], centroid[1])]
        # Candidate angles
        cand_angles = [0, 30]
        
        for cx, cy in centers:
            for ang in cand_angles:
                S = required_side(all_verts, cx, cy, ang)
                if S < best_S:
                    best_S = S
                    best_cx = cx
                    best_cy = cy
                    best_angle = ang
        
        # Also try a small local search around the best one found
        # This might be too slow, so we'll skip it for now and rely on the grid
        
        return best_S, best_cx, best_cy, best_angle

    def required_side(verts, cx, cy, angle_deg):
        """Calculate the side length of a regular hexagon centered at (cx, cy) with angle angle_deg
        that contains all given vertices."""
        rad = np.radians(angle_deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        
        # The hexagon is defined by 6 linear inequalities:
        # n_k . (p - c) <= S * h_k
        # where n_k are the normals to the sides, and h_k is the apothem (inradius)
        # For a regular hexagon of side S, the inradius r = S * sqrt(3)/2
        # The normals are at angles rad + 30 + 60*k for k=0..5
        # Wait, the normals to the sides of a flat-top hexagon (vertex at angle 0)
        # are at 30, 90, 150, 210, 270, 330 degrees.
        # So for a rotated hexagon, normals are at rad + 30 + 60*k
        
        normals = []
        for k in range(6):
            norm_ang = rad + np.radians(30) + np.radians(60) * k
            normals.append([np.cos(norm_ang), np.sin(norm_ang)])
        normals = np.array(normals)
        
        # For each normal n, the constraint is:
        # n . (p - c) <= S * (sqrt(3)/2)
        # So S >= 2/sqrt(3) * max_k (n_k . (p - c))
        
        max_proj = 0.0
        for p in verts:
            rel = p - np.array([cx, cy])
            projs = normals @ rel
            max_proj = max(max_proj, np.max(projs))
        
        # Note: max_proj can be negative if the hexagon is very large and centered far away,
        # but since we're finding the minimal S, we assume the hexagon is tight enough.
        # Actually, we need max over all vertices and all normals.
        # The formula S >= 2/sqrt(3) * max(n . (p-c)) is correct.
        
        S = max_proj * 2.0 / SQRT3
        # S must be at least the inradius to be a valid hexagon, but usually S will be positive.
        if S < 0:
            S = 0.0 # This shouldn't happen for reasonable inputs
        
        return S

    # Optimization strategy
    # 1. Generate a good initial guess using a lattice
    # 2. Use differential evolution to optimize positions and discrete angles
    
    # For differential evolution, we need to encode the discrete angles.
    # We can use the first 12*2 = 24 dimensions for positions, and the next 12 for angles.
    # But differential evolution works on continuous spaces.
    # We can map continuous values to discrete angles by rounding.
    
    def decode(params):
        pos = params[:N*2].reshape(N, 2)
        ang_cont = params[N*2:N*2+N]
        # Map continuous angle values to discrete indices 0-5
        # Let's say the range is [0, 6], where integer values correspond to the angles
        ang_idx = np.clip(np.round(ang_cont), 0, 5).astype(int)
        return pos, ang_idx

    def objective_de(params):
        pos, ang_idx = decode(params)
        if not is_valid(pos, ang_idx):
            return 1e6
        S, _, _, _ = outer_side_length(pos, ang_idx)
        return S

    # Initial guess: hexagonal lattice
    # Hexagonal lattice spacing: distance between centers should be at least sqrt(3)
    # Let's use a lattice with spacing slightly larger than sqrt(3) to ensure validity
    # Actually, for unit hexagons, the minimum distance between centers is sqrt(3)
    # if they are aligned. With rotations, it can be smaller.
    
    # Let's start with a grid of 4x3 or 3x4 hexagons
    # A 3x4 grid of hexagons in a hexagonal lattice
    # Coordinates in a hexagonal lattice:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's try to place 12 hexagons in a hexagonal cluster
    # A hexagon of side length 2 (in terms of hexagon units) has 1+6+12 = 19 hexagons
    # A hexagon of side length 1 has 7 hexagons.
    # We need 12, so let's take a partial cluster.
    
    # Let's use a 4x3 rectangular grid in the hexagonal lattice
    # q ranges from -1 to 2, r ranges from 0 to 2
    # But we need to pick 12 points.
    
    # Let's just use a simple grid and let the optimizer fix it.
    # Initial positions: 4x3 grid
    pos_init = np.zeros((N, 2))
    for i in range(N):
        q = i % 4
        r = i // 4
        pos_init[i, 0] = SQRT3 * (q + r/2.0)
        pos_init[i, 1] = 1.5 * r
    
    # Center the grid
    pos_init -= np.mean(pos_init, axis=0)
    
    # Initial angles: all 0
    ang_init = np.zeros(N)
    
    params_init = np.concatenate([pos_init.flatten(), ang_init])
    
    # Bounds for differential evolution
    # Positions: -10 to 10
    # Angles: 0 to 6 (will be rounded to 0-5)
    bounds = [(-10, 10)] * (N*2) + [(0, 6)] * N
    
    # Run differential evolution
    best_S = float('inf')
    best_params = None
    
    try:
        res = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-4,
            polish=False, # We'll polish later
            init='sobol',
            workers=1
        )
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # If differential evolution didn't find anything good, try the initial guess
    if best_params is None or best_S > 1e5:
        # Check if initial guess is valid
        if is_valid(pos_init, np.zeros(N, dtype=int)):
            S, _, _, _ = outer_side_length(pos_init, np.zeros(N, dtype=int))
            if S < best_S:
                best_S = S
                best_params = np.concatenate([pos_init.flatten(), np.zeros(N)])
        else:
            # Try to find a valid configuration by spreading out
            # This is a fallback, we'll just return a large hexagon
            for i in range(N):
                pos_init[i, 0] = i * 3.0
                pos_init[i, 1] = 0.0
            if is_valid(pos_init, np.zeros(N, dtype=int)):
                S, _, _, _ = outer_side_length(pos_init, np.zeros(N, dtype=int))
                if S < best_S:
                    best_S = S
                    best_params = np.concatenate([pos_init.flatten(), np.zeros(N)])
    
    # Polish the best solution using local optimization
    if best_params is not None:
        # We'll use L-BFGS-B to polish the positions, keeping angles fixed
        # or allow small changes in angles too
        
        pos, ang_idx = decode(best_params)
        
        def objective_polish(params):
            pos_p = params[:N*2].reshape(N, 2)
            ang_p = params[N*2:N*2+N]
            ang_idx_p = np.clip(np.round(ang_p), 0, 5).astype(int)
            if not is_valid(pos_p, ang_idx_p):
                return 1e6
            S, _, _, _ = outer_side_length(pos_p, ang_idx_p)
            return S
        
        # Bounds for polishing
        polish_bounds = [(-15, 15)] * (N*2) + [(-0.5, 6.5)] * N
        
        try:
            res2 = minimize(
                objective_polish,
                best_params,
                method='L-BFGS-B',
                bounds=polish_bounds,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
        except Exception as e:
            pass
    
    # Extract the final configuration
    pos, ang_idx = decode(best_params)
    
    # Calculate the final outer hexagon parameters
    S, cx, cy, angle = outer_side_length(pos, ang_idx)
    
    # If S is still very large or invalid, return a fallback
    if S > 100:
        # Fallback: spread out hexagons
        for i in range(N):
            pos[i, 0] = i * 3.0
            pos[i, 1] = 0.0
        ang_idx = np.zeros(N, dtype=int)
        S, cx, cy, angle = outer_side_length(pos, ang_idx)
    
    # Construct the return values
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(angles_list[ang_idx[i]])])
    
    return inner, (float(cx), float(cy)), float(S), float(angle)

# EVOLVE-BLOCK-END
